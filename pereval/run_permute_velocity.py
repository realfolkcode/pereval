import os
import argparse
import numpy as np
import symusic
import partitura
from symusic import Score, Note
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_dir', type=str, required=True, help='path to evaluation dir')
    parser.add_argument('--output_dir', type=str, required=True, help='path to output directory')

    args = parser.parse_args()

    evaluation_dir = args.evaluation_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    for i, score_name in enumerate(tqdm(os.listdir(evaluation_dir))):
        score_dir = os.path.join(evaluation_dir, score_name)
        if not os.path.isdir(score_dir):
            continue

        output_score_dir = os.path.join(output_dir, score_name)
        os.makedirs(output_score_dir, exist_ok=True)
        
        perf_name_lst = []

        for perf_name in os.listdir(score_dir):
            if not perf_name.endswith(".mid"):
                continue
            perf_name_lst.append(perf_name)

        perf_name_lst.append(perf_name_lst[0])

        for i in range(len(perf_name_lst) - 1):
            perf_name_a = perf_name_lst[i]
            perf_name_b = perf_name_lst[i + 1]

            perf_path_a = os.path.join(score_dir, perf_name_a)
            perf_path_b = os.path.join(score_dir, perf_name_b)

            mid_a = Score(perf_path_a)
            mid_b = Score(perf_path_b)

            notes_a = mid_a.tracks[0].notes
            notes_b = mid_b.tracks[0].notes

            notes_a = notes_a.sort(lambda x: (x.pitch, x.time))
            notes_b = notes_b.sort(lambda x: (x.pitch, x.time))

            notes_a = notes_a.numpy()
            notes_b = notes_b.numpy()

            for pitch in range(128):
                indices_a = np.argwhere(notes_a['pitch'] == pitch)
                indices_b = np.argwhere(notes_b['pitch'] == pitch)

                min_len = min(len(indices_a), len(indices_b))
                indices_a = indices_a[:min_len]
                indices_b = indices_b[:min_len]

                notes_a['velocity'][indices_a] = notes_b['velocity'][indices_b]
            
            mid_a.tracks[0].notes = Note.from_numpy(**notes_a, ttype=mid_a.ttype)

            output_path = os.path.join(output_score_dir, perf_name_a)
            mid_a.dump_midi(output_path)


if __name__ == "__main__":
    main()
