# Pereval: Performance Evaluation toolkit

This is an official repository for the paper "Integrating Contextual Embeddings into Evaluation of Expressive MIDI Piano Performances" (**ISMIR 2026**).

Authors: Dmitrii Gavrilev, Ilya Borovik, and Vladimir Viro

**NOTE:** The documentation and requirements are still being prepared.

## Repository structure:

- `/pereval/` contains the main utilities (such as alignment, feature extraction, calculation of attribute-scoped and deep feature metrics) used in our research paper
- `/scripts/` contains the scripts used to reproduce the experiments in the paper
- `/examples/` contains the notebooks for data analysis and examples of usage

**Scripts**
- `run_save_alignment.py` - calculates the note alignment using Parangonar
- `run_correlation.py` - calculates attribute-scoped correlations between sets of performances
- `run_kde.py` - calculates Kernel Density Estimators for attribute-scoped KL-divergence
- `run_embed_aria.py` - extracts Aria embeddings
- `run_shift_notes.py` - shifts (truncates the first few) notes in performances
- `run_permute_velocity.py` - transfers the velocities of the notes between performances

**Notebooks:**
- `mahalanobis_pseudo_ratings.ipynb` - comparing Mahalanobis-based performance pseudo-ratings with human MOS from the listening test 
