<div align="center">

# Pereval: Piano Performance Evaluation toolkit

## Integrating Contextual Embeddings into Evaluation of Expressive MIDI Piano Performances

[![arXiv](https://img.shields.io/badge/arXiv-2607.27909-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2607.27909)
[![PyPI](https://img.shields.io/pypi/v/pereval)](https://pypi.org/project/pereval/)

</div>

**Dmitrii Gavrilev, Ilya Borovik, and Vladimir Viro**

This is an official repository for the paper ["Integrating Contextual Embeddings into Evaluation of Expressive MIDI Piano Performances"](https://arxiv.org/pdf/2607.27909) (**Accepted at ISMIR 2026**).

[Documentation (WIP)](https://realfolkcode.github.io/pereval/)

## Overview

In this paper, we tackle the problem of evaluating and comparing sets of piano MIDI performances.

Traditional metrics (which we coin **attribute-scoped**), such as *Pearson correlation* and *reconstruction error*, that are most commonly used in Expressive Performance Rendeering (EPR) have several drawbacks. Namely, they are limited by a single expressive attribute, such as tempo, articulation, and dynamics of a performance, disregarding other musical aspects and context. Second, they often require a note-level alignment between a score and a performance, hindering the scalability of evaluation.

We adapt recently proposed kernel-based methods from image and audio generation domains for EPR. **Deep feature metrics**, *Kernel Music Distance* (KMD) and *Kernel Performance Distance* (KPD), leverage rich contextual features from Aria and CLaMP3 symbolic music understanding models. The adapted metrics are scalable (do not require alignments) and contextual.

In short, this project aims to go beyond the traditional metrics for expressive MIDI piano performances and unify them under one toolkit to simplify evaluation routines for MIR practitioners.

|                          | Fidelity | Diversity | Alignment-free | Score-aware | Contextual | Per-sample |
|--------------------------|:--------:|:---------:|:--------------:|:-----------:|:----------:|:----------:|
| Reconstruction error     | ✅       | ❌        | ❌             | ✅          | ❌         | ✅         |
| Inter-correlation        | ✅       | ❌        | ❌             | ✅          | ❌         | ✅         |
| Intra-correlation        | ❌       | ✅        | ❌             | ✅          | ❌         | ❌         |
| KL Divergence            | ✅       | ✅        | ✅             | ❌          | ❌         | ❌         |
| Fréchet Music Distance   | ✅       | ✅        | ✅             | ❌          | ✅         | ✅         |
| Kernel Music Distance    | ✅       | ✅        | ✅             | ❌          | ✅         | ✅         |
| Kernel Performance Distance | ✅    | ✅        | ✅             | ✅          | ✅         | ✅         |
| Mahalanobis Distance     | ✅       | ❌        | ✅             | ✅          | ✅         | ✅         |
| Relative Mahalanobis Distance | ✅  | ❌        | ✅             | ✅          | ✅         | ✅         |
| Marginal Mahalanobis Distance | ✅  | ❌        | ✅             | ❌          | ✅         | ✅         |

*Comparison of attribute-scoped and deep feature metrics for evaluating MIDI piano performances.*

## Installation

1. Prerequisites:
- Python >= 3.12 
- [torch](https://pytorch.org/) (only versions >=2.3 are supported officially)
- [Aria](https://github.com/EleutherAI/aria)
2. Install pereval:
```sh
pip install -U pereval
```

## Repository structure:

- `/pereval/` contains the main utilities (such as alignment, feature extraction, calculation of attribute-scoped and deep feature metrics) used in our research paper
- `/examples/` contains the notebooks for data analysis and examples of usage

**Command-line interface:**
Please refer to the [documentation](https://realfolkcode.github.io/pereval/cli.html) for examples of CLI usage.

**Notebooks:**
- `mahalanobis_pseudo_ratings.ipynb` - comparing Mahalanobis-based Aria pseudo-ratings with human MOS from the listening test 

## Acknowledgements

We would like to thank the authors of [FADtk](https://github.com/microsoft/fadtk), [KADtk](https://github.com/YoonjinXD/kadtk), [Aria](https://github.com/EleutherAI/aria), [CLaMP3](https://github.com/sanderwood/clamp3), [Parangonar](https://github.com/sildater/parangonar), and [symusic](https://github.com/Yikai-Liao/symusic) for their open source contributions in music and audio research.

## Citation

If you use this package in your research, please consider citing:

```bibtex
@inproceedings{gavrilev2026integrating,
    title     = {Integrating Contextual Embeddings into Evaluation of Expressive MIDI Piano Performances},
    author    = {Gavrilev, Dmitrii and Borovik, Ilya and Viro, Vladimir},
    year      = 2026,
    booktitle = {Proceedings of the 27th International Society for Music Information Retrieval Conference (ISMIR)},
}
```
