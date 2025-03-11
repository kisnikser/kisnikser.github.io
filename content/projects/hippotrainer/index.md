---
title: "HippoTrainer: Gradient-Based Hyperparameter Optimization for PyTorch"
date: 2025-03-11
lastmod: 2025-03-11
tags: ["hyperparameter optimization", "PyTorch", "Optuna"]
author: ["Daniil Dorin", "Igor Ignashin", "Nikita Kiselev", "Andrey Veprikov"]
description:
summary: "We release a Python library for gradient-based hyperparameter optimization, implementing cutting-edge algorithms that leverage automatic differentiation to efficiently tune hyperparameters."
editPost:
  URL: https://github.com/intsystems/hippotrainer
  Text: GitHub
showToc: true
showReadingTime: true
---

In this blog-post we present our Python library [HippoTrainer](https://github.com/intsystems/hippotrainer) (or `hippotrainer`) for gradient-based hyperparameter optimization, implementing cutting-edge algorithms that leverage automatic differentiation to efficiently tune hyperparameters.

## Introduction <a name="introduction"></a>

Hyperparameter tuning is time-consuming and computationally expensive, often requiring extensive trial and error to find optimal configurations. There is a variety of hyperparameter optimization methods, such as Grid Search, Random Search, Bayesian Optimization, etc. In the case of continuous hyperparameters, the gradient-based methods arise.

We implemented four effective and popular methods in one package, leveraging the unified, simple and clean structure. Below we delve into the problem statement and methods description.

### Hyperparameter Optimization Problem

Given a vector of model parameters $\mathbf{w} \in \mathbb{R}^P$ and a vector of hyperparameters $\boldsymbol{\lambda} \in \mathbb{R}^H$. One aims to find optimal hyperparameters $\boldsymbol{\lambda}^*$, solving the bi-level optimization problem:

$$
\begin{aligned}
&\boldsymbol{\lambda}^\* = \argmin\_{\boldsymbol{\lambda}} \mathcal{L}\_{\text{val}}(\mathbf{w}^\*, \boldsymbol{\lambda}), \\\
\text{s.t. } &\mathbf{w}^\* = \argmin\_{\mathbf{w}} \mathcal{L}\_{\text{train}}(\mathbf{w}, \boldsymbol{\lambda})
\end{aligned}
$$

Often $\mathbf{w}$ are optimized with gradient descent, so **unrolled optimization** is typically used:

$$
\mathbf{w}\_{t+1} = \boldsymbol{\Phi}(\mathbf{w}\_{t}, \boldsymbol{\lambda}), \quad t = 0, \ldots, T-1.
$$

Typical way to optimize continuous hyperparameters is the **gradient-based optimization** that involves automatic differentiation through this unrolled optimization formula.

### Hypergradient Calculation

Chain rule gives us a hypergradient $d\_{\boldsymbol{\lambda}} \mathcal{L}\_{\text{val}}(\mathbf{w}\_T, \boldsymbol{\lambda})$, viewing $\mathbf{w}\_T$ as a function of $\boldsymbol{\lambda}$:
$$
    \underbrace{d\_{\boldsymbol{\lambda}} \mathcal{L}\_{\text{val}}(\mathbf{w}\_T, \boldsymbol{\lambda})}\_{\text{hypergradient}} = \underbrace{\nabla\_{\boldsymbol{\lambda}} \mathcal{L}\_{\text{val}}(\mathbf{w}\_T, \boldsymbol{\lambda})}\_{\text{hyperparam direct grad.}} + \underbrace{\nabla\_{\mathbf{w}} \mathcal{L}\_{\text{val}}(\mathbf{w}\_T, \boldsymbol{\lambda})}\_{\text{parameter direct grad.}} \times \underbrace{\frac{d\mathbf{w}\_T}{d\boldsymbol{\lambda}}}\_{\text{\textbf{best-response Jacobian}}}
$$

- Here **best-response Jacobian** is hard to compute!

Typical Solution — Implicit Function Theorem:
$$
    \frac{d\mathbf{w}\_T}{d\boldsymbol{\lambda}} = - \underbrace{\left[ \nabla^2\_{\mathbf{w}} \mathcal{L}\_{\text{train}}(\mathbf{w}\_T, \boldsymbol{\lambda}) \right]^{-1}}\_{\text{\textbf{inverted} training Hessian}} \times \underbrace{\nabla\_{\mathbf{w}} \nabla\_{\boldsymbol{\lambda}} \mathcal{L}\_{\text{train}} (\mathbf{w}\_T, \boldsymbol{\lambda})}\_{\text{training mixed partials}}.
$$

- Hessian **inversion** is a cornerstone of many algorithms.

The next section contains information about each of the methods presented in our library, as they can be generalized to solve the above problem in different ways.

## Methods

To exactly invert a $P \times P$ Hessian, we require $\mathcal{O}(P^3)$ operations, which is intractable for modern NNs. We can efficiently approximate the inverse with the Neumann series:
$$
  \left[ \nabla^2\_{\mathbf{w}} \mathcal{L}\_{\text{train}}(\mathbf{w}\_T, \boldsymbol{\lambda}) \right]^{-1} = \lim\_{i \to \infty} \sum\_{j=0}^{i} \left[ \mathbf{I} - \nabla^2\_{\mathbf{w}} \mathcal{L}\_{\text{train}} (\mathbf{w}\_T, \boldsymbol{\lambda}) \right]^j.
$$

Using different numbers of terms in this series, one can derive a list of methods.

### T1 – T2 ([Luketina et al. 2015](https://arxiv.org/abs/1511.06727))

In this method, the number of terms $i$ equals $0$, and the number of inner optimization steps $T$ is equal to $1$. Therefore, this method is also named Greedy gradient-based hyperparameter optimization. In particular, here we have:
$$
  \left[ \nabla^2\_{\mathbf{w}} \mathcal{L}\_{\text{train}}(\mathbf{w}\_T, \boldsymbol{\lambda}) \right]^{-1} \approx \mathbf{I}.
$$

### IFT ([Lorraine et al. 2019](https://arxiv.org/abs/1911.02590))

Another method uses a pre-determined number of terms in the Neumann series. It also efficiently computes $\nabla\_{\boldsymbol{\lambda}} \mathcal{L}\_{\text{val}}(\mathbf{w}\_T, \boldsymbol{\lambda}) \times \left[ \nabla^2\_{\mathbf{w}} \mathcal{L}\_{\text{train}}(\mathbf{w}\_T, \boldsymbol{\lambda}) \right]^{-1}$, leveraging the following approximation formula:
$$
  \left[ \nabla^2\_{\mathbf{w}} \mathcal{L}\_{\text{train}}(\mathbf{w}\_T, \boldsymbol{\lambda}) \right]^{-1} \approx \sum\_{j=0}^{i} \left[ \mathbf{I} - \nabla^2\_{\mathbf{w}} \mathcal{L}\_{\text{train}} (\mathbf{w}\_T, \boldsymbol{\lambda}) \right]^j.
$$

### HOAG ([Pedregosa, 2016](https://arxiv.org/abs/1602.02355))

In contrast to the previous ones, this method solves the linear system using the Conjugate Gradient to invert the Hessian approximately. The following system is solved w.r.t. $\mathbf{z}$:
$$
\nabla^2\_{\mathbf{w}} \mathcal{L}\_{\text{train}}(\mathbf{w}\_T, \boldsymbol{\lambda}) \cdot \mathbf{z} = \nabla\_{\boldsymbol{\lambda}} \mathcal{L}\_{\text{val}}(\mathbf{w}\_T, \boldsymbol{\lambda}).
$$

### DrMAD ([Fu et al. 2016](https://arxiv.org/abs/1601.00917))

The last method in our package is not straightforward. Instead of storing all intermediate weights $\mathbf{w}\_0, \ldots, \mathbf{w}\_T$, it approximates the training trajectory as a linear combination of the initial $\mathbf{w}\_0$ and final $\mathbf{w}\_T$ weights:
$$
  \mathbf{w}(\beta) = (1 - \beta) \mathbf{w}\_0 + \beta \mathbf{w}\_T, \quad 0 < \beta < 1.
$$
Then it uses such approximation to perform the backward pass on the hyperparameters.

## Implementation (see our [GitHub](https://github.com/intsystems/hippotrainer) for details)

TODO

## Demo

TODO

## Conclusion

TODO

## References

[1] Luketina et al. ["Scalable Gradient-Based Tuning of Continuous Regularization Hyperparameters"](https://arxiv.org/abs/1511.06727). arXiv preprint arXiv:1511.06727 (2015).

[2] Lorraine et al. ["Optimizing Millions of Hyperparameters by Implicit Differentiation"](https://arxiv.org/abs/1911.02590). arXiv preprint arXiv:1911.02590 (2019).

[3] Pedregosa. ["Hyperparameter optimization with approximate gradient"](https://arxiv.org/abs/1602.02355). arXiv preprint arXiv:1602.02355 (2016).

[4] Fu et al. ["DrMAD: Distilling Reverse-Mode Automatic Differentiation for Optimizing Hyperparameters of Deep Neural Networks"](https://arxiv.org/abs/1601.00917). arXiv preprint arXiv:1601.00917 (2016).