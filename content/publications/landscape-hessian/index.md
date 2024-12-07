---
title: "Unraveling the Hessian: A Key to Smooth Convergence in Loss Function Landscapes"
date: 2024-08-20
lastmod: 2024-12-07
tags: ["neural networks", "loss function landscape", "Hessian matrix", "convergence analysis", "image classification"]
author: ["Nikita Kiselev", "Andrey Grabovoy"]
description: 
summary: "This paper explore the convergence of the loss landscape in neural networks as the sample size increases, focusing on the Hessian matrix to understand the local geometry of the loss function."
cover:
    image: "losses_difference.png"
    alt: "Overview"
    relative: false
editPost:
    URL:
    Text:

---

---

##### Links

+ [arXiv](https://arxiv.org/abs/2409.11995) 
+ [Code](https://github.com/kisnikser/landscape-hessian)

---

##### Abstract

The loss landscape of neural networks is a critical aspect of their training, and understanding its properties is essential for improving their performance. In this paper, we investigate how the loss surface changes when the sample size increases, a previously unexplored issue. We theoretically analyze the convergence of the loss landscape in a fully connected neural network and derive upper bounds for the difference in loss function values when adding a new object to the sample. Our empirical study confirms these results on various datasets, demonstrating the convergence of the loss function surface for image classification tasks. Our findings provide insights into the local geometry of neural loss landscapes and have implications for the development of sample size determination techniques.

---

##### Figure 1: Overview

![](losses_difference.png)

<!-- ---

##### Citation

```BibTeX
@article{dorin2024forecastingfmriimages,
  author = {Dorin, Daniil and Kiselev, Nikita and Grabovoy, Andrey and Strijov, Vadim},
  journal = {Health Information Science and Systems},
  number = {1},
  pages = {55},
  title = {Forecasting fMRI images from video sequences: linear model analysis},
  volume = {12},
  year = {2024}
}
``` -->

<!-- ---

##### Related material

+ [Presentation slides](presentation1.pdf)
+ [Summary of the paper](https://www.penguinrandomhouse.com/books/110403/unusual-uses-for-olive-oil-by-alexander-mccall-smith/) -->
