# Pereval: Piano Performance Evaluation toolkit

## Integrating Contextual Embeddings into Evaluation of Expressive MIDI Piano Performances

**Dmitrii Gavrilev, Ilya Borovik, and Vladimir Viro**

This is an official repository for the paper "Integrating Contextual Embeddings into Evaluation of Expressive MIDI Piano Performances" (**Accepted by ISMIR 2026**).

**NOTE:** The paper will be available by the end of July/start of August. The documentation and requirements are still being prepared.

## Overview

In this paper, we tackle the problem of evaluating and comparing sets of piano MIDI performances.

Traditional metrics (which we coin **attribute-scoped**), such as *Pearson correlation* and *reconstruction error*, that are most commonly used in Expressive Performance Rendeering (EPR) have several drawbacks. Namely, they are limited by a single expressive attribute, such as tempo, articulation, and dynamics of a performance, disregarding other musical aspects and context. Second, they often require a note-level alignment between a score and a performance, hindering the scalability of evaluation.

We adapt recently proposed kernel-based methods from image and audio generation domains for EPR. **Deep feature metrics**, *Kernel Music Distance* (KMD) and *Kernel Performance Distance* (KPD), leverage rich contextual features from Aria and CLaMP3 symbolic music understanding models. The adapted metrics are scalable (do not require alignments) and contextual.

In short, this project aims to go beyond the traditional metrics for expressive MIDI piano performances and unify them under one toolkit to simplify evaluation routines for MIR practitioners.

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
- `mahalanobis_pseudo_ratings.ipynb` - comparing Mahalanobis-based Aria pseudo-ratings with human MOS from the listening test 

## Acknowledgements

We would like to thank the authors of [FADtk](https://github.com/microsoft/fadtk), [KADtk](https://github.com/YoonjinXD/kadtk), [Aria](https://github.com/EleutherAI/aria), [CLaMP3](https://github.com/sanderwood/clamp3), and [Parangonar](https://github.com/sildater/parangonar) for their open source contributions in music and audio research.
