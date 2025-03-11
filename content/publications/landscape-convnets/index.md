---
title: "ConvNets Landscape Convergence: Hessian-Based Analysis of Matricized Networks"
date: 2024-12-12
lastmod: 2025-03-11
tags:
  [
    "convolutional neural network",
    "loss function",
    "Hessian matrix",
    "convergence rate",
    "sample size determination",
  ]
author: ["Vladislav Meshkov", "Nikita Kiselev", "Andrey Grabovoy"]
description:
summary: "This paper introduces a method for estimating the Hessian matrix norm in convolutional neural networks, offering insights into the loss landscape's local behaviour, supported by empirical convergence analysis."
editPost:
  URL: "https://doi.org/10.1109/ISPRAS64596.2024.10899113"
  Text: "2024 Ivannikov Ispras Open Conference (ISPRAS)"
---

---

##### Links

- [IEEEXplore](https://ieeexplore.ieee.org/document/10899113)
- [ResearchGate](https://www.researchgate.net/publication/389459624_ConvNets_Landscape_Convergence_Hessian-Based_Analysis_of_Matricized_Networks)
- [Code](https://github.com/Drago160/Hessian-Based-Analysis-of-Matricized-Networks)

---

##### Abstract

The Hessian of a neural network is an important aspect for understanding the loss landscape and the characteristic of network architecture. The Hessian matrix captures important information about the curvature, sensitivity, and local behavior of the loss function. Our work proposes a method that enhances the understanding of the local behavior of the loss function and can be used to analyze the behavior of neural networks and also for interpreting the parameters in these networks. In this paper, we consider an approach to investigate the properties of the deep neural network, using the Hessian. We propose a method for estimating the Hessian matrix norm for a specific type of neural networks like convolutional. We have obtained the results for both 1D and 2D convolutions, as well as for the fully connected head in these networks. Our empirical analysis supports these findings, demonstrating convergence in the loss function landscape. We have evaluated the Hessian norm for neural networks represented as a product of matrices and considered how this estimate affects the landscape of the loss function.

---

##### Citation

```BibTeX
@inproceedings{meshkov2024convnets,
  title={ConvNets Landscape Convergence: Hessian-Based Analysis of Matricized Networks},
  author={Meshkov, Vladislav and Kiselev, Nikita and Grabovoy, Andrey},
  booktitle={2024 Ivannikov Ispras Open Conference (ISPRAS)},
  pages={1--10},
  year={2024},
  organization={IEEE}
}
```
