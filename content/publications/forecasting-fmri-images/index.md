---
title: "Forecasting fMRI Images From Video Sequences: Linear Model Analysis"
date: 2024-04-04
tags: ["time series forecasting", "fmri images", "linear models", "correlation analysis"]
author: ["Daniil Dorin", "Nikita Kiselev", "Andrey Grabovoy"]
description: 
summary: "Using linear models and testing hypotheses about the relationship between data" 
cover:
  image: "publications/forecasting-fmri-images/scheme.png"
  alt: ""
  relative: true
---

---

### Links

- [📝 Paper](https://github.com/DorinDaniil/Forecasting-fMRI-Images/blob/main/paper/main.pdf) 
- [</> Code](https://github.com/DorinDaniil/Forecasting-fMRI-Images/tree/main/code)

---

### Abstract

The issue of reconstructing the relationship between functional magnetic resonance imaging (fMRI) sensor readings and human perception of the external is investigated. The study analyzes the dependence between the fMRI images and the videos viewed by individuals. Based on this analysis, a method is proposed for approximating the fMRI readings using the video sequence. The method is based on the assumption that there is a time-invariant hemodynamic response to changes in blood oxygen levels. A linear model is constructed for each individual voxel in the fMRI image, assuming that the image sequence follows a Markov property. To test the proposed method, a computational experiment was conducted on a dataset collected during tomographic examinations of a large number of individuals. The performance of the method was evaluated based on the experimental data, and hypotheses were tested regarding the invariance of the model weights and the correctness of the method.

---

