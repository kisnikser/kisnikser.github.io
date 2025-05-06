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

Hyperparameter tuning remains one of the most laborious and resource-intensive aspects of machine learning development. Traditional methods like grid search, random search, or Bayesian optimization often demand exhaustive iterations to identify optimal configurations, especially for high-dimensional problems. However, when hyperparameters are **continuous** (e.g., regularization coefficients, learning rates, or architectural parameters), gradient-based optimization offers a compelling alternative by leveraging automatic differentiation to compute hypergradients—gradients of validation loss with respect to hyperparameters.

In this post, we introduce **HippoTrainer**, a PyTorch-native library that democratizes access to state-of-the-art gradient-based hyperparameter optimization (HPO) methods. Our library implements four powerful algorithms: T1-T2, IFT, HOAG, and DrMAD, that efficiently approximate the inverse Hessian required for hypergradient computation, enabling scalable optimization even for large neural networks. By abstracting these complex techniques into a unified, user-friendly interface inspired by PyTorch's `Optimizer`, HippoTrainer bridges the gap between theoretical advances in HPO and practical implementation.

### The Bi-Level Optimization Framework

At the core of gradient-based HPO lies the bi-level optimization problem:

$$
\begin{aligned}
&\boldsymbol{\lambda}^\* = \argmin\_{\boldsymbol{\lambda}} \mathcal{L}\_{\text{val}}(\mathbf{w}^\*, \boldsymbol{\lambda}), \\\
\text{s.t. } &\mathbf{w}^\* = \argmin\_{\mathbf{w}} \mathcal{L}\_{\text{train}}(\mathbf{w}, \boldsymbol{\lambda})
\end{aligned}
$$

Here, $\boldsymbol{\lambda}$ represents hyperparameters (e.g., L2 regularization strength), and $\mathbf{w}$ denotes model parameters. The inner loop minimizes training loss to produce optimal weights $\mathbf{w}^*$, while the outer loop adjusts $\boldsymbol{\lambda}$ to minimize validation loss. Gradient-based methods differentiate through the inner optimization process to compute hypergradients $\frac{d\mathcal{L}\_{\text{val}}}{d\boldsymbol{\lambda}}$, enabling direct optimization of $\boldsymbol{\lambda}$ via gradient descent.

### Key Challenges and HippoTrainer's Solutions

Computing hypergradients requires solving:
$$
\frac{d\mathcal{L}\_{\text{val}}}{d\boldsymbol{\lambda}} = \nabla\_{\boldsymbol{\lambda}} \mathcal{L}\_{\text{val}} + \nabla\_{\mathbf{w}} \mathcal{L}\_{\text{val}} \cdot \underbrace{\frac{d\mathbf{w}}{d\boldsymbol{\lambda}}}_{\text{Best-Response Jacobian}}
$$

The **best-response Jacobian** $\frac{d\mathbf{w}}{d\boldsymbol{\lambda}}$ involves inverting the Hessian $\nabla^2\_{\mathbf{w}} \mathcal{L}\_{\text{train}}$, a computationally prohibitive operation for large models. HippoTrainer tackles this with approximations like Neumann series expansions (IFT), conjugate gradient methods (HOAG), or trajectory simplifications (DrMAD), making HPO tractable for real-world applications.

In the following sections, we explore these methods and demonstrate how HippoTrainer streamlines their implementation, empowering developers to focus on model innovation rather than optimization logistics.

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

You can use our library to tune almost all (see below) hyperparameters in your own code.
The `HyperOptimizer` interface is very similar to [`Optimizer`](https://pytorch.org/docs/2.6/optim.html#torch.optim.Optimizer) from `PyTorch`.

It supports key functionalities:
1. `step` to do an optimization step over parameters (or hyperparameters, see below)
2. `zero_grad` to zero out the parameters gradients (same as `optimizer.zero_grad()`)

We provide demo experiments with each implemented method in [this notebook](https://github.com/intsystems/hippotrainer/blob/main/notebooks/demo.ipynb).
They works as follows:
1. Get next batch from train dataloader
2. Forward and backward on calculated loss
3. `hyper_optimizer.step(loss)` do model parameters step and (if inner steps were accumulated) hyperparameters step (calculate hypergradients, do the optimization step, zeroes hypergradients)
4. `hyper_optimizer.zero_grad()` zeroes the model parameters gradients (same as `optimizer.zero_grad()`)

### `Optimizer` vs. `HyperOptimizer` method `step`

Gradient-based hyperparameters optimization involves hyper-optimization steps during the
model parameters optimization. Thus, we combine `Optimizer` method `step` with `inner_steps`,
defined by each method.

For example, `T1T2` do NOT use any inner steps, therefore optimization over parameters
and hyperparameters is done step by step. But `Neumann` method do some inner optimization steps
over model parameters before it do the hyperstep.

See more details [here](https://github.com/intsystems/hippotrainer/blob/60cbafd6614bf057e83268da6cebf04ae2e6d7e7/src/hippotrainer/hyper_optimizer.py#L121).

### Supported hyperparameters types

The `HyperOptimizer` logic is well-suited for almost all **CONTINUOUS** (required for gradient-based methods) hyperparameters types:
1. Model hyperparameters (e.g., gate coefficients)
2. Loss hyperparameters (e.g., L1/L2-regularization)

However, it currently does **not** support (or support, but actually was not sufficiently tested) **learning rate tuning**.
We plan to improve our functionality in future releases, stay tuned!

## Conclusion

HippoTrainer introduces a powerful toolkit for gradient-based hyperparameter optimization in `PyTorch`, bridging the gap between cutting-edge research and practical application. By implementing advanced methods like T1-T2, IFT, HOAG, and DrMAD, our library enables efficient tuning of continuous hyperparameters—such as regularization coefficients or model-specific parameters—through automatic differentiation and Hessian approximations. This approach drastically reduces the computational burden compared to traditional grid or random search, while offering scalability for modern neural networks.

Our `HyperOptimizer` interface abstracts the complexity of these methods, providing a familiar, `PyTorch`-like API that integrates seamlessly into existing workflows. With support for customizable inner optimization steps and pre-built demos, users can experiment with hyperparameter tuning with minimal code changes. While current limitations include the exclusion of learning rate optimization (a focus of future work), HippoTrainer lays a robust foundation for gradient-driven hyperparameter discovery.

We invite researchers and practitioners to explore HippoTrainer on [GitHub](https://github.com/intsystems/hippotrainer), where you can test the methods firsthand, contribute improvements, or adapt the framework to new use cases. By democratizing access to efficient hyperparameter optimization, we aim to accelerate progress in machine learning model development.

## References

[1] Luketina et al. ["Scalable Gradient-Based Tuning of Continuous Regularization Hyperparameters"](https://arxiv.org/abs/1511.06727). arXiv preprint arXiv:1511.06727 (2015).

[2] Lorraine et al. ["Optimizing Millions of Hyperparameters by Implicit Differentiation"](https://arxiv.org/abs/1911.02590). arXiv preprint arXiv:1911.02590 (2019).

[3] Pedregosa. ["Hyperparameter optimization with approximate gradient"](https://arxiv.org/abs/1602.02355). arXiv preprint arXiv:1602.02355 (2016).

[4] Fu et al. ["DrMAD: Distilling Reverse-Mode Automatic Differentiation for Optimizing Hyperparameters of Deep Neural Networks"](https://arxiv.org/abs/1601.00917). arXiv preprint arXiv:1601.00917 (2016).