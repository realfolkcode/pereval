import parangonar as pa
import partitura
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List


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
