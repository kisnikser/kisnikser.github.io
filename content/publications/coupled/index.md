---
title: "Decentralized Optimization with Coupled Constraints"
date: 2024-05-15
tags: ["decentralized optimization", "convex optimization", "affine constraints", "coupled constraints"]
author: ["Demyan Yarmoshik", "Dmitry Kovalev", "Alexander Rogozin", "Nikita Kiselev", "Daniil Dorin", "Alexander Gasnikov"]
description: 
summary: "In this paper, we propose the first first-order algorithm with accelerated linear convergence for decentralized optimization with coupled constraints" 
cover:
  image: #"publications/coupled/coupled.pdf"
  alt: ""
  relative: true
---

---

### Abstract

We consider the decentralized minimization of a separable objective $\sum_{i=1}^{n} f_i(x_i)$, where the variables are coupled through an affine constraint $\sum_{i=1}^n\left(A_i x_i - b_i\right) = 0$. We assume that the functions $f_i$, matrices $A_i$, and vectors $b_i$ are stored locally by the nodes of a computational network, and that the functions $f_i$ are smooth and strongly convex.

This problem has significant applications in resource allocation and systems control, yet the known complexity bounds for existing algorithms fall short when compared to closely related consensus optimization methods. In this paper, we propose the first first-order algorithm with accelerated linear convergence for decentralized optimization with coupled constraints. We demonstrate the practical performance of our algorithm on a vertical federated learning task.

---

