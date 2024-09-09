---
title: "Unraveling the Hessian: A Key to Smooth Convergence in Loss Function Landscapes"
date: 2024-08-20
tags: ["neural networks", "loss function landscape", "Hessian matrix", "convergence analysis", "image classification"]
author: ["Nikita Kiselev", "Andrey Grabovoy"]
description: 
summary: "Exploring the convergence of the loss landscape in neural networks as the sample size increases, focusing on the Hessian matrix to understand the local geometry of the loss function" 
cover:
  image: "publications/landscape-hessian/losses_difference.pdf"
  alt: ""
  relative: true
---

---

### Links

- [📝 Paper](https://github.com/kisnikser/landscape-hessian/blob/main/paper/main.pdf) 
- [</> Code](https://github.com/kisnikser/landscape-hessian/tree/main/code)

---

### Abstract

The loss landscape of neural networks is a critical aspect of their training, and understanding its properties is essential for improving their performance. In this paper, we investigate how the loss surface changes when the sample size increases, a previously unexplored issue. We theoretically analyze the convergence of the loss landscape in a fully connected neural network and derive upper bounds for the difference in loss function values when adding a new object to the sample. Our empirical study confirms these results on various datasets, demonstrating the convergence of the loss function surface for image classification tasks. Our findings provide insights into the local geometry of neural loss landscapes and have implications for the development of sample size determination techniques.

---

