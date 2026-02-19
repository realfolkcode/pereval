import parangonar as pa
import partitura
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List

import miditok
import torch
import time
import symusic
from parangonar import THEGLUENOTE_CHECKPOINT
from parangonar.match import TheGlueNote
from parangonar.match.matchers import get_note_matches_with_updating_map
from parangonar.match.gluenote_utils import (
    get_shifted_and_stretched_and_agnostic_midis,
    minimal_note_array_from_symusic,
    note_array_to_symusic_score,
    format_score_performance_alignment,
    format_note_array_alignment,
    DEFAULT_NOTE,
    get_local_path_from_confidence_matrix,
    get_input_to_ref_map,
)


class NoteAligner:
    def load_score_perf_alignment(self,
                                  alignment_path: str,
                                  perf,
                                  score) -> NDArray:
        """Loads alignment between score and performance.
        
        Args:
            alignment_path: The path to .match file.
            perf: Performance note array.
            score: Score note array.
        
        Returns:
            An array with score indices in the first row,
            and performance indices in the second row.
        """
        alignment = partitura.io.importmatch.load_matchfile(filename=alignment_path)
        alignment = partitura.io.importmatch.alignment_from_matchfile(alignment)
        alignment = list(filter(lambda elem: elem['label'] == 'match',
                                alignment))
        
        score_indices = [self._parse_note_id(pair['score_id']) for pair in alignment]
        perf_indices = [self._parse_note_id(pair['performance_id']) for pair in alignment]

        score_idx_map = dict(zip([self._parse_note_id(elem) for elem in score['id']],
                                 list(range(len(score['id'])))))
        perf_idx_map = dict(zip([self._parse_note_id(elem) for elem in perf['id']],
                                list(range(len(perf['id'])))))
        
        score_indices = [score_idx_map[elem] for elem in score_indices]
        perf_indices = [perf_idx_map[elem] for elem in perf_indices]

        return np.vstack((score_indices, perf_indices))
    
    def _parse_note_id(self,
                       note_id):
        return int(str(note_id).split('-')[0][1:])

    def get_onset_indices(self,
                          score,
                          indices = None) -> List[NDArray]:
        """Retrieves onset indices from score.

        Args:
            score: Score note array.
            indices: A subset of score note indices.

        Returns:
            A list of onset note indices grouped together.
        """
        onsets = score['onset_div']
        if indices is not None:
            onsets = onsets[indices]
        
        onset_indices = np.split(np.argsort(onsets),
                                 np.cumsum(np.unique(onsets, return_counts=True)[1])[:-1])
        return onset_indices


class TheGlueNoteMatcher(object):
    def __init__(self):
        self.prepare_model()
        self.tokenizer = miditok.Structured()
        self.unmatched_idx = 100000000
        self.matching_threshold = (0.5,)

    def prepare_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(
            THEGLUENOTE_CHECKPOINT, map_location=torch.device(self.device)
        )
        self.model = TheGlueNote(device=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, score_note_array, performance_note_array):
        midi_score = note_array_to_symusic_score(score_note_array)
        midi_performance = note_array_to_symusic_score(performance_note_array)
        alignment = self.run_model_inference(
            midi_performance,
            midi_score,
            performance_note_array=performance_note_array,
            score_note_array_full=score_note_array,
            return_formatted_alignment=True,
        )
        return alignment

    def run_model_inference(
        self,
        performance_midi,
        score_midi,
        performance_note_array=None,
        score_note_array_full=None,
        return_formatted_alignment=False,
    ):
        alignment = self.get_dtw_alignment_from_model(
            encoder_model=self.model,
            tokenizer=self.tokenizer,
            input_midi1=score_midi,
            input_midi2=performance_midi,
            unmatched_idx=self.unmatched_idx,
        )

        if (
            return_formatted_alignment
            and score_note_array_full is not None
            and performance_note_array is not None
        ):
            alignment = format_score_performance_alignment(
                score_note_array_full,
                performance_note_array,
                alignment,
                unmatched_idx=self.unmatched_idx,
            )
        return alignment

    def get_dtw_alignment_from_model(
        self,
        encoder_model,
        tokenizer,
        input_midi1,
        input_midi2,
        unmatched_idx=100000000,
    ):
        # setup and preprocessing of files
        sequence_length = encoder_model.position_number - 1
        input_midi2, input_midi1 = get_shifted_and_stretched_and_agnostic_midis(
            input_midi2, input_midi1
        )
        note_array = minimal_note_array_from_symusic(input_midi1)
        note_array_ref = minimal_note_array_from_symusic(input_midi2)
        tokens1 = tokenizer(input_midi1)
        tokens2 = tokenizer(input_midi2)
        sample = {
            "s1": np.array(tokens1[0].ids, dtype=int),
            "s2": np.array(tokens2[0].ids, dtype=int),
        }

        no_notes_s1 = len(note_array)
        no_notes_s2 = len(note_array_ref)
        index_shift = int(sequence_length / 2)
        no_slices_s1 = no_notes_s1 // (index_shift) + 1
        no_slices_s2 = no_notes_s2 // (index_shift) + 1
        full_similarity_matrix = np.zeros((no_notes_s1, no_notes_s2))

        # loop over windows
        for i in range(no_slices_s1):
            for j in range(no_slices_s2):
                current_idx_1 = [i * index_shift, (i + 2) * index_shift]
                current_idx_2 = [j * index_shift, (j + 2) * index_shift]

                # pad and prepare the note sequences
                s1 = sample["s1"][current_idx_1[0] * 4 : current_idx_1[1] * 4]
                s2 = sample["s2"][current_idx_2[0] * 4 : current_idx_2[1] * 4]
                s1_matrix_end = sequence_length
                s2_matrix_end = sequence_length

                if len(s1) < sequence_length * 4:
                    # pad the sequence
                    padding_len1 = sequence_length * 4 - len(s1)
                    padding1 = np.array(
                        DEFAULT_NOTE * int(padding_len1 / 4)
                    )  # np.zeros(padding_len1)
                    s1_matrix_end = int(len(s1) / 4)
                    s1 = np.concatenate((s1, padding1)).astype(int)

                if len(s2) < sequence_length * 4:
                    # pad the sequence
                    padding_len2 = sequence_length * 4 - len(s2)
                    padding2 = np.array(
                        DEFAULT_NOTE * int(padding_len2 / 4)
                    )  # np.zeros(padding_len2)
                    s2_matrix_end = int(len(s2) / 4)
                    s2 = np.concatenate((s2, padding2)).astype(int)

                s1 = np.concatenate((np.zeros(4), s1)).astype(int)
                s2 = np.concatenate((np.zeros(4), s2)).astype(int)
                sequences = (
                    torch.from_numpy(np.concatenate((s1, s2)).astype(int))
                    .contiguous()
                    .unsqueeze(0)
                )

                # call the encoder
                sequences = sequences.to(encoder_model.device)
                confidence_matrix = encoder_model(
                    sequences, return_confidence_matrix=True
                )
                confidence_matrix_segment = (
                    confidence_matrix[0, 1 : s1_matrix_end + 1, 1 : s2_matrix_end + 1]
                    .detach()
                    .cpu()
                    .numpy()
                )

                len1 = current_idx_1[1] - current_idx_1[0]
                len2 = current_idx_2[1] - current_idx_2[0]

                if confidence_matrix_segment.shape[0] < len1:
                    len1 = confidence_matrix_segment.shape[0]
                if confidence_matrix_segment.shape[1] < len2:
                    len2 = confidence_matrix_segment.shape[1]

                full_similarity_matrix[
                    current_idx_1[0] : current_idx_1[0] + len1,
                    current_idx_2[0] : current_idx_2[0] + len2,
                ] += confidence_matrix_segment

        now = time.time()
        (
            path,
            starting_path,
            ending_path,
            s1_exclusion_start,
            s1_exclusion_end,
        ) = get_local_path_from_confidence_matrix(full_similarity_matrix)
        then = time.time()
        print("DTW local path time: ", then - now)
        print("PATH length:", len(path), full_similarity_matrix.shape)

        s1_to_s2_map = get_input_to_ref_map(
            note_array, note_array_ref, path, return_callable=False
        )

        alignment = get_note_matches_with_updating_map(
            note_array,
            note_array_ref,
            s1_to_s2_map,
            onset_threshold=1000,
            unmatched_idx=unmatched_idx,
        )

        return alignment

    def match_midi(self, midi_path_0, midi_path_1):
        midi_0 = symusic.Score(midi_path_0)
        midi_1 = symusic.Score(midi_path_1)
        alignment = self.run_model_inference(
            midi_0, midi_1, return_formatted_alignment=False
        )
        return alignment
