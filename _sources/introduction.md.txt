# Introduction

In this project, we tackle the problem of evaluating and comparing sets of piano MIDI performances.

Traditional metrics (which we coin **attribute-scoped**), such as *Pearson correlation* and *reconstruction error*, that are most commonly used in Expressive Performance Rendeering (EPR) have several drawbacks. Namely, they are limited by a single expressive attribute, such as tempo, articulation, and dynamics of a performance, disregarding other musical aspects and context. Second, they often require a note-level alignment between a score and a performance, hindering the scalability of evaluation.

We adapt recently proposed kernel-based methods from image and audio generation domains for EPR. **Deep feature metrics**, *Kernel Music Distance* (KMD) and *Kernel Performance Distance* (KPD), leverage rich contextual features from Aria and CLaMP3 symbolic music understanding models. The adapted metrics are scalable (do not require alignments) and contextual.

In short, this project aims to go beyond the traditional metrics for expressive MIDI piano performances and unify them under one toolkit to simplify evaluation routines for MIR practitioners.
