---
title: "Forecasting fMRI Images From Video Sequences: Linear Model Analysis"
date: 2024-04-04
tags: ["time series forecasting", "fmri images", "linear models", "correlation analysis"]
author: ["Daniil Dorin", "Nikita Kiselev", "Andrey Grabovoy"]
description: 
summary: "Exploring the correlation between video sequences and fMRI images, using a linear autoregressive model" 
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

Over the past few decades, a variety of significant scientific breakthroughs have been achieved in the fields of brain encoding and decoding using the functional magnetic resonance imaging (fMRI). Many studies have been conducted on the topic of human brain reaction to visual stimuli. However, the relationship between fMRI images and video sequences viewed by humans remains complex and is often studied using large transformer models. In this paper, we investigate the correlation between videos presented to participants during an experiment and the resulting fMRI images. To achieve this, we propose a method for creating a linear model that predicts changes in fMRI signals based on video sequence images. A linear model is constructed for each individual voxel in the fMRI image, assuming that the image sequence follows a Markov property. Through the comprehensive qualitative experiments, we demonstrate the relationship between the two time series. We hope that our findings contribute to a deeper understanding of the human brain's reaction to external stimuli and provide a basis for future research in this area.

---

