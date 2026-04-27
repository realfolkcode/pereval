# Pereval: Performance Evaluation toolkit

This source code is part of the anonymized supplementary material for the ISMIR 2026 submission paper "Integrating Contextual Embeddings into Evaluation of Expressive MIDI Piano Performances".

Please do not redistribute. This code is only available as part of the reviewing process.

**NOTE:** The documentation and requirements are not available in the current version. We will reformat and release the more accessible version upon the acceptance.

## Repository structure:

- `/pereval/` contains the main utilities (such as alignment, feature extraction, calculation of attribute-scoped and deep feature metrics) used in our research paper

**Scripts**
- `run_save_alignment.py` - calculates the note alignment using Parangonar
- `run_correlation.py` - calculates attribute-scoped correlations between sets of performances
- `run_kde.py` - calculates Kernel Density Estimators for attribute-scoped KL-divergence
- `run_embed_aria.py` - extracts Aria embeddings
- `run_shift_notes.py` - shifts (truncates the first few) notes in performances
- `run_permute_velocity.py` - transfers the velocities of the notes between performances
Other scripts are miscellaneous.

**Notebooks:**
- We will make the data analysis notebooks available upon the acceptance. 
