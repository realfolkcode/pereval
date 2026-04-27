import numpy as np
from numpy.typing import NDArray
from typing import Dict, List


def calculate_correlation(score,
                          real_perf,
                          gen_perf,
                          real_score_perf_indices,
                          gen_score_perf_indices,
                          note_aligner,
                          left_percentage: float | None = None,
                          right_percentage: float | None = None) -> Dict[str, float]:
    """Calculates correlation between two performances.
    
    Args:
        real_perf: Real performance.
        gen_perf: Generated performance
        real_score_perf_indices: Score-to-performance alignment
          for real performance.
        gen_score_perf_indices: Score-to-performance alignment
          for generated performance.
        note_aligner: Note aligner.
        left_percentage: The left fraction of notes from 0 to 1.
        right_percentage: The right fraction of notes from 0 to 1.
    
    Returns:
        A dictionary with attribute correlations.
    """
    score_indices, real_score_indices, gen_score_indices = np.intersect1d(real_score_perf_indices[0],
                                                                          gen_score_perf_indices[0],
                                                                          return_indices=True)

    if left_percentage is not None:
        assert len(score_indices) == len(real_score_indices)
        assert len(score_indices) == len(gen_score_indices)

        start_score = int(left_percentage * len(score_indices))
        end_score = int(right_percentage * len(score_indices))
        score_indices = score_indices[start_score:end_score]

        start_real = int(left_percentage * len(real_score_indices))
        end_real = int(right_percentage * len(real_score_indices))
        real_score_indices = real_score_indices[start_real:end_real]

        start_gen = int(left_percentage * len(gen_score_indices))
        end_gen = int(right_percentage * len(gen_score_indices))
        gen_score_indices = gen_score_indices[start_gen:end_gen]

    real_perf_indices = real_score_perf_indices[1][real_score_indices]
    gen_perf_indices = gen_score_perf_indices[1][gen_score_indices]

    onset_indices = note_aligner.get_onset_indices(score, indices=score_indices)

    real_onsets = real_perf['onset_sec'][real_perf_indices]
    real_onsets = aggregate(onset_indices, real_onsets)
    real_ioi = real_onsets[1:] - real_onsets[:-1]

    gen_onsets = gen_perf['onset_sec'][gen_perf_indices]
    gen_onsets = aggregate(onset_indices, gen_onsets)
    gen_ioi = gen_onsets[1:] - gen_onsets[:-1]

    real_velocity = real_perf['velocity'][real_perf_indices]
    gen_velocity = gen_perf['velocity'][gen_perf_indices]

    real_duration = real_perf['duration_sec'][real_perf_indices]
    gen_duration = gen_perf['duration_sec'][gen_perf_indices]

    ioi_corr = np.corrcoef(real_ioi, gen_ioi)[0, 1]
    velocity_corr = np.corrcoef(real_velocity, gen_velocity)[0, 1]
    duration_corr = np.corrcoef(real_duration, gen_duration)[0, 1]
    
    return {'ioi': ioi_corr,
            'velocity': velocity_corr,
            'duration': duration_corr}


def aggregate(indices_lst: List[NDArray],
              arr: NDArray):
    """Aggregates means of an array grouped by indices.
    
    Args:
        indices_lst:
    """
    n = len(indices_lst)
    
    all_indices = np.concatenate(indices_lst)
    values = arr[all_indices]
    
    labels = np.concatenate([
        np.full(len(indices), i, dtype=int) 
        for i, indices in enumerate(indices_lst)
    ])
    
    counts = np.bincount(labels, minlength=n)
    sums = np.bincount(labels, weights=values, minlength=n)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        means = sums / counts    
    return means
