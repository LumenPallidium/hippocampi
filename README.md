# Hippocampi

## Introduction

This repository contains implementations of computational models of the hippocampus.

Specifically, it contains an implementation of the architecture Vector-HaSH from the paper [High-capacity flexible hippocampal associative and episodic memory enabled by prestructured “spatial” representations](https://www.biorxiv.org/content/10.1101/2023.11.28.568960v1), which itself relies on [continuous attractor networks (CANs)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000291) and [Hebbian rules for learning pseudoinverses](https://arxiv.org/abs/1207.3368).

Note that while [High-capacity flexible hippocampal associative and episodic memory enabled by prestructured “spatial” representations](https://www.biorxiv.org/content/10.1101/2023.11.28.568960v1) cites [continuous attractor networks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000291), it is actually based on a simpler, discrete grid version of the CAN. Both the original (src/continuous_attractors) and the simpler version (src/grids) are implemented here.

## Organization

* The file src/continuous_attractors.py contains a grid cell simulator, using the continuous attractor model. It plots a video (see figures) of a simulated "rat" moving around in a box.
* The file src/grids.py contains the implementation of the Vector-HaSH architecture and the main "recall" method of the model.
