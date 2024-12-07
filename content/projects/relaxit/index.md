---
title: "Just Relax It! Leveraging relaxation for discrete variables optimization"
date: 2024-12-07
lastmod: 2024-12-07
tags: ["Relaxation", "Gumbel-Softmax", "Straight-Through Estimator", "Python", "Library", "Package", "PyTorch", "Pyro"]
author: ["Daniil Dorin", "Igor Ignashin", "Nikita Kiselev", "Andrey Veprikov"]
description:
summary: "We release a cutting-edge Python library designed to streamline the optimization of discrete probability distributions in neural networks, offering a suite of advanced relaxation techniques compatible with PyTorch."
cover:
    image: "overview.png"
    alt: "Overview"
    relative: false
editPost:
    URL: 
    Text:
showToc: true
---

![Overview](overview.png)

In this blog-post we present our Python library ["Just Relax It"](https://github.com/intsystems/discrete-variables-relaxation) (or `relaxit`) designed to streamline the optimization of discrete probability distributions in neural networks, offering a suite of advanced relaxation techniques compatible with PyTorch.

## Introduction

Recent development of generative models, e.g. VAE and Diffusion Models, has driven relevant mathematical tools. 
Any generative model contains some source of randomness to make new objects.
This randomness represents a certain probability distribution, from which random variables are sampled.
Thus, the task of training a generative model is often comes down to optimization of such distribution parameters.

Pioneering generative models work with **continous** distributions like Normal one.
However, for some modalities, e.g. texts or graphs, it is quite natural to use **discrete** distributions – Bernoulli, Categorical, etc.

Thus, we present our new Python library ["Just Relax It"](https://github.com/intsystems/discrete-variables-relaxation) that combines the best techniques for relaxing discrete distributions (we will explain what that means later) into an easy-to-use package. And it is compatible with PyTorch!

We start with a basic example that shows how parameter optimization typically happens for continuous distributions, then we move on smoothly to the case of discrete distributions. After that, we talk about the main relaxation techniques used in our library and make a demo of training a VAE with discrete latent variables.

### VAE example

{{< figure src="demo.png" title="Fig. 1. Variational Autoencoder (VAE) architecture." >}}

The original VAE ([Kingma & Welling, 2014](https://arxiv.org/abs/1312.6114)) consists of two parts:

1. Encoder $q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})$, which is represented by a neural network $g_{\boldsymbol{\phi}}(\mathbf{x})$ that outputs parameters of the latent Gaussian distribution;
2. Decoder $p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})$, which is represented by a neural network $f_{\boldsymbol{\theta}}(\mathbf{z})$ that outputs parameters of the sample distribution (typically Gaussian or Bernoulli).

The math behind training a VAE is not obvious actually, so we will just focus on the ELBO (evidence lower bound), which needs to be maximized w.r.t. the parameters of the encoder and decoder:
$$
\mathcal{L}_{\boldsymbol{\phi}, \boldsymbol{\theta}}(\mathbf{x}) = \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})} \log p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z}) - KL(q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) \to \max_{\boldsymbol{\phi}, \boldsymbol{\theta}}. 
$$

During the **M-step**, we gonna derive the unbiased estimator for the gradient $\nabla_{\boldsymbol{\theta}}\mathcal{L}_{\boldsymbol{\phi}, \boldsymbol{\theta}}(\mathbf{x})$:

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}}\mathcal{L}_{\boldsymbol{\phi}, \boldsymbol{\theta}}(\mathbf{x})
&= \textcolor{blue}{\nabla_{\boldsymbol{\theta}}} \int q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) \log p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z}) d\mathbf{z} \\
&= \int q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) \textcolor{blue}{\nabla_{\boldsymbol{\theta}}} \log p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z}) d\mathbf{z} \\
&\approx \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z}^*), \quad \mathbf{z}^* \sim q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}),
\end{aligned}
$$

where the last approximation is a Monte-Carlo sampling estimator.

However, on the **E-step** it is quite tricky to get unbiased estimator for the gradient $$\nabla_{\boldsymbol{\phi}}\mathcal{L}_{\boldsymbol{\phi}, \boldsymbol{\theta}}(\mathbf{x}).$$ 

As density function $q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})$ depends on the parameters $\boldsymbol{\phi}$, it is impossible to use the Monte-Carlo estimation:

$$
\begin{aligned}
\nabla_{\boldsymbol{\phi}}\mathcal{L}_{\boldsymbol{\phi}, \boldsymbol{\theta}}(\mathbf{x})
&= \textcolor{blue}{\nabla_{\boldsymbol{\phi}}} \int q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) \log p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z}) d\mathbf{z} - \nabla_{\boldsymbol{\phi}} KL(q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) \\
&\textcolor{red}{\neq} \int q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) \textcolor{blue}{\nabla_{\boldsymbol{\theta}}} \log p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z}) d\mathbf{z} - \nabla_{\boldsymbol{\phi}} KL(q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})), \\
\end{aligned}
$$

and this is the moment where the **reparameterization trick** arises, we reparameterize the outputs of the **encoder**:

$$
\begin{aligned}
\nabla_{\boldsymbol{\phi}} \int \textcolor{blue}{q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})} \log p_{\boldsymbol{\theta}}(\mathbf{x}|\textcolor{OliveGreen}{\mathbf{z}}) d\mathbf{z}
&= \int \textcolor{blue}{p(\boldsymbol{\epsilon})} \nabla_{\boldsymbol{\phi}} \log p_{\boldsymbol{\theta}}(\mathbf{x}|\textcolor{OliveGreen}{\mathbf{g}_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon})}) d\boldsymbol{\epsilon} \\
&\approx \nabla_{\boldsymbol{\phi}} \log p_{\boldsymbol{\theta}}(\mathbf{x}|\textcolor{OliveGreen}{\boldsymbol{\sigma}_{\boldsymbol{\phi}}(\mathbf{x})} \odot \textcolor{blue}{\boldsymbol{\epsilon}^*} + \textcolor{OliveGreen}{\boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x})}), \quad \textcolor{blue}{\boldsymbol{\epsilon}^*} \sim \mathcal{N}(0, \mathbf{I}),
\end{aligned}
$$

so we move the randomness to the $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$, and use the deterministic transform $$\mathbf{z} = \mathbf{g}_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon})$$ in order to get unbiased gradient. It also needs to be mentioned that normal assumptions for $q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})$ and $p(\mathbf{z})$ allows us to compute $KL$ analytically and thus calculate the gradient $\nabla_{\boldsymbol{\phi}}KL(q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$.

The above example gives us an understanding of a crucial reparameterization trick, which allows us to get unbiased gradient estimations for the continuous latent space in VAE model. But actually discrete representations $\mathbf{z}$ are potentially a more natural fit for many of the modalities (like texts or images), which moves us to the **discrete VAE latentes**. Therefore

- Our encoder should output discrete distribution;
- We need the analogue of the reparameterization trick for the discrete distribution;
- Our decoder should input discrete random variable.

The classical solution for the discrete variables reparameterization trick is **Gumbel-Softmax** ([Jang et al. 2017](https://arxiv.org/abs/1611.01144)) or **Concrete relaxation** ([Maddison et al. 2017](https://arxiv.org/abs/1611.00712)).

#### Gumbel distribution

$$ g \sim \mathrm{Gumbel}(0, 1) \quad \Leftrightarrow \quad g = -\log( - \log u), \quad u \sim \mathrm{Uniform}[0, 1] $$

#### Theorem (Gumbel-Max trick)

Let $g_k \sim \mathrm{Gumbel}(0, 1)$ for $k = 1, \ldots, K$. Then a discrete random variable

$$ c = \arg\max_k [\log \pi_k + g_k] $$

has a categorical distribution $c \sim \mathrm{Categorical}(\boldsymbol{\pi})$.

- We could sample from the discrete distribution using Gumbel-Max reparameterization;
- Here parameters and random variable sampling are separated (reparameterization trick);
- **Problem:** we still have non-differentiable $\arg\max$ operation.

#### Gumbel-Softmax relaxation

$$ \hat{\mathbf{c}} = \mathrm{softmax}\left( \frac{\log q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) + \mathbf{g}}{\tau} \right) $$

Here $\tau$ is a temperature parameters. Now we have differentiable operation, but the gradient estimator is biased now. However, if $\tau \to 0$, then the estimation becomes more and more accurate.

### Other relaxation methods

So far, we have talked about one possible example of a discrete variable relaxation (VAE with discrete latent variables) and the classical approach to solving this problem. However, the Gumbel-Softmax relaxation was actually proposed a long time ago. There are actually many other relaxation techniques that can provide more flexible and accurate (and even unbiased) gradient estimates. The rest of our blog-post will focus on cutting-edge relaxation techniques and how we built a Python library that uses them, which works with the PyTorch framework to train neural networks efficiently.

## Algorithms

In this section, we provide a short description for each of the implemented methods. We can generalize them as follows. Suppose that $x$ is a random variable, $f$ if a function (say, the loss function), and we are interested in computing $\frac{\partial}{\partial \theta} \mathbb{E}_{x}\left[ f(x) \right]$. It is quite natural decision because typical ML problem looks like this. So, two different ideas exist:

- *Score function* (SF) estimator. In this case, we are given a parameterized probability distribution $x \sim p(\cdot; \theta)$ and use

$$ \frac{\partial}{\partial \theta} \mathbb{E}_{x}\left[ f(x) \right] = \mathbb{E}_{x} \left[ f(x) \frac{\partial}{\partial \theta} \log p(x; \theta) \right]. $$

- *Pathwise derivative* (PD) estimator. In this case $x$ is a determinisitc, differentiable function of $\theta$ and another random variable $z$, i.e. we can write $x(z, \theta)$:

$$ \frac{\partial}{\partial \theta} \mathbb{E}_{x}\left[ f(x(z, \theta)) \right] = \mathbb{E}_{z} \left[ \frac{\partial}{\partial \theta} f(x(z, \theta)) \right]. $$

The latter one we have seen previously in the VAE example! A sample $x$ from $\mathcal{N}(\mu, \sigma^2)$ can be obtained by sampling $z$ from the standard normal distribution $\mathcal{N}(0, 1)$ and then transforming it using $x(z, \theta) = \sigma z + \mu$. And this is called reparameterization trick.

However, when $x$ is a discrete variable, it is quite tricky to make a pathwise derivative estimator, i.e. to reparameterize the discrete distribution. And this is the moment of relaxation! We replace $x$ with a continuous relaxation $x(z, \theta) \approx x_{\tau}(z, \theta)$, where $\tau > 0$ is a temperature that controls the tightness of the relaxaton (at low temperatues, the relaxation is nearly high).

### Relaxed Bernoulli ([Yamada, Lindenbaum et al. 2018](https://arxiv.org/abs/1810.04247))

The reparameterization trick is inspired by the idea of stochastic gates and aims to approximate a Bernoulli random variable in a more relaxed manner. This technique involves drawing a random variable, denoted as $\epsilon$, from a normal distribution with a mean of 0 and a variance of $\sigma^2$, where $\sigma$ is a fixed parameter. The random variable $\epsilon$ is then used to compute $z$ as follows:

$$
\begin{aligned}
\epsilon &\sim \mathcal{N}(0, \sigma^2),\\
z &= \min (1, \max (0, \mu + \epsilon)),
\end{aligned}
$$

where $\mu$ is a learnable parameter that can be tuned during the training process. This transformation ensures that the resulting $z$ value is bounded between 0 and 1, thereby **relaxing the Bernoulli distribution**.

### Correlated relaxed Bernoulli ([Lee, Imrie et al. 2022](https://openreview.net/pdf?id=oDFvtxzPOx))

This method generates correlated gate vectors from a multivariate Bernoulli distribution using a Gaussian copula:

$$C_R(U_1, \ldots, U_p) = \Phi_R(\Phi^{-1}(U_1), \ldots, \Phi^{-1}(U_p)),$$

where $\Phi_R$ is the joint CDF of a multivariate Gaussian distribution with correlation matrix $R$, and $\Phi^{-1}$ is the inverse CDF of the standard univariate Gaussian distribution.

The gate vector $m$ is generated as:

$$m_k =
\begin{cases}
1, & \text{if } U_k \leq \pi_k, \\
0, & \text{if } U_k > \pi_k,
\end{cases}\quad k = 1, \ldots, p,
$$

where $U_k$ are correlated random variables preserving the input feature correlations.

For differentiability, a continuous relaxation is applied:

$$m_k = \sigma \left( \frac{1}{\tau} \left( \log \frac{U_k}{1 - U_k} + \log \frac{\pi_k}{1 - \pi_k} \right) \right),$$

where $\sigma(x) = \frac{1}{1 + \exp(-x)}$ is the sigmoid function, and $\tau$ is a temperature hyperparameter. Thus, the **Bernolli distribution relaxes**.

### Gumbel-Softmax TOP-K ([Kool et al. 2019](https://arxiv.org/pdf/1903.06059))

Suppose we want to get $K$ samples without replacement (i.e., not repeating) according to the Categorical distribution with probabilities $\boldsymbol{\pi}$. Similar to the Gumbel-Max method,  let $g_k \sim \mathrm{Gumbel}(0, 1)$ for $k = 1, \ldots, K$, then the Gumbel-Max-Top$K$ Theorem says, that the values of the form

$$c_1, \ldots , c_K = \text{Arg}\underset{k}{\text{top}}K [ \log\pi_k + g_k]$$

have the $\mathrm{Categorical}(\boldsymbol{\pi})$ distribution without replacement.

This approach has all the same pros and cons as the classical Gumbel-Max trick, however, they can be fixed with the Gumbel-Softmax relaxation using a simple loop:

$$ Algorithm $$

Therefore, this method allows us to **relax the Categorical distribution**.

### Stochastic Times Smooth ([Bengio et al. 2013](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=62c76ca0b2790c34e85ba1cce09d47be317c7235))

The Stochastic Times Smooth distribution can be written as follows

$$
\begin{aligned}
p_i &= \sigma(a_i), \\
b_i &\sim \text{Binomial}(\sqrt{p_i}), \\
h_i &= b_i \sqrt{p_i},
\end{aligned}
$$

where $a_i$ is a parameter of this distribution. This one provides a **Bernoulli distribution relaxation**.

### Invertible Gaussian ([Potapczynski et al. 2019](https://arxiv.org/abs/1912.09588))

The idea of this method is to remove interpretability of parameters in Gumbel-Softmax relaxation, and achieve then higher quality. Namely, the goal of Gumbel-Softmax relaxation is to relax $\mathbf{z} \sim \mathrm{Cat}(\boldsymbol{\pi})$ proposing temperature parameter $\tau \to 0$, which concentrates mass on vertices:

$$\tilde{\mathbf{z}} = \mathrm{softmax}\left(\frac{\log{\boldsymbol{\pi}} + \mathbf{G}}{\tau}\right),$$

where $G_i \sim \mathrm{Gumbel}(0, 1)$.

The authors propose an alternative family of distributions that works by transforming Gaussian noise $\boldsymbol{\epsilon}$ through invertible transformation onto the simplex. In particular, map $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ to simplex, using invertible  $g(\cdot, \tau)$ with temperature $\tau$:

$$
\begin{aligned}
\mathbf{y} &= \boldsymbol{\mu} + \mathrm{diag}(\boldsymbol{\sigma}) \boldsymbol{\epsilon},\\
\tilde{\mathbf{z}} &= g(\mathbf{y}, \tau) = \mathrm{softmax}_{++}(\mathbf{y/\tau})
\end{aligned}
$$

Thus, this is one more **relaxation of Categorical distribution**.

### Hard Concrete ([Louizos et al. 2018](https://arxiv.org/abs/1712.01312)

The relaxed Bernoulli method can be viewed from another angle, if we consider it in the following form:

$$
\begin{aligned}
s &\sim q(s | \phi),\\
z &= \min (1, \max (0, s)),
\end{aligned}
$$

where the distribution $q(s | \phi)$ is normal $\mathcal{N}(\mu, \sigma^2)$. The idea of Hard Concrete is to 1) consider a Gumbel-Softmax relaxation $q(s | \phi) = \mathrm{GS}(s | \phi)$ with parameters $\phi = (\log \alpha, \tau)$; 2) stretch it from $(0, 1)$ to the wider interval $(\gamma, \zeta)$, with $\gamma < 0$ and $\zeta > 1$; and then 3) apply a hard-sigmoid on its random samples.

$$
\begin{aligned}
s &= \sigma\left( (g + \log \alpha) / \tau \right), \quad g \sim \mathrm{Gumbel}(0, 1),\\
\bar{s} &= s (\zeta - \gamma) + \gamma,\\
z &= \min (1, \max (0, \bar{s})).
\end{aligned}
$$

This distribution provides a **Bernoulli variable relaxation**, applying hard-sigmoid technique to make two delta peaks at zero and one.

### Closed-form Laplace Bridge ([Hobbhahn et al. 2020](https://arxiv.org/abs/2003.01227))

In this and the next sections we consider quite another approaches used for discrete variables, but not relaxation actually. This one, closed-form Laplace Bridge, is an approach of approximating Dirichlet distribution with Logistic-Normal, and vice versa.

Why should we consider it? Indeed, these two distributions lies on the simplex and it is natural decision to find the parameters to match each of them with particular one.

In particular, the analytic map from the Dirichlet distribution parameter $\boldsymbol{\alpha} \in \mathbb{R}_{+}^{K}$ to the parameters of the Gaussian $\boldsymbol{\mu} \in \mathbb{R}^{K}$ and symmetric positive definite $\boldsymbol{\Sigma} \in \mathbb{R}^{K \times K}$ is given by

$$
\begin{aligned}
\mu_i &= \log \alpha_i - \frac{1}{K} \sum_{k=1}^{K} \log \alpha_k,\\
\Sigma_{ij} &= \delta_{ij} \frac{1}{\alpha_i} - \frac{1}{K} \left( \frac{1}{\alpha_i} + \frac{1}{\alpha_j} - \frac{1}{K} \sum_{k=1}^{K} \frac{1}{\alpha_k} \right),
\end{aligned}
$$

and the pseudo-inverse of this one, which maps the Gaussian parameters to those of the Dirichlet as

$$
\alpha_k = \frac{1}{\Sigma_{kk}} \left( 1 - \frac{2}{K} + \frac{e^{\mu_k}}{K^2} \sum_{l=1}^{K} e^{-\mu_l} \right).
$$

And this is what is called **Laplace Bridge between Dirichlet and Logistic-Normal distributions**.

## Implementation (see our [GitHub]([](https://github.com/intsystems/relaxit)) for details)

In this section we describe our package design. The most famous Python probabilistic libraries with a built-in differentiation engine are [PyTorch](https://pytorch.org/docs/stable/index.html) and [Pyro](https://docs.pyro.ai/en/dev/index.html). Thus, we implement the `relaxit` library consistently with both of them. Specifically, we

1. Take a base class for PyTorch-compatible distributions with Pyro support `TorchDistribution`, for which we refer to [this page](https://docs.pyro.ai/en/dev/distributions.html#torchdistribution) on documentation.
2. Inherent each of the considered relaxed distributions from this `TorchDistribution`.
3. Implement `batch_shape` and `event_shape` properties that defines the distribution samples shapes.
4. Implement `rsample()` and `log_prob()` methods as key two of the proposed algorithms. These methods are responsible for sample with reparameterization trick and log-likelihood computing respectively.

For closed-form Laplace Bridge between Dirichlet and Logistic-Normal distributions we extend the base PyTorch KL-divergence method with one more realization. We also implement a `LogisticNormalSoftmax` distribution, which is a transformed distribution from the `Normal` one. In contrast to original `LogisticNormal` from Pyro or PyTorch, this one uses `SoftmaxTransform`, instead of `StickBreakingTransform` that allows us to remain in the same dimensionality.

## Demo

Our demo code is available at [this link](https://github.com/intsystems/discrete-variables-relaxation/tree/main/demo). For demonstration purposes, we divide our algorithms in three[^1] different groups. Each group relates to the particular experiment:

1. Laplace Bridge between Dirichlet and Logistic-Normal distributions;
2. REINFORCE;
3. Other relaxation methods.

[^1]: We also implement REINFORCE algorithm as a score function estimator alternative for our relaxation methods that are inherently pathwise derivative estimators. This one is implemented only for demo experiments and is not included into the source code of package.

**Laplace Bridge.** This part relates to the demonstation of closed-form Laplace Bridge between Dirichlet and Logistic-Normal distributions. We subsequently 1) initialize a Dirichlet distribution with random parameters; 2) approximate it with a Logistic-Normal distribution; 3) approximate obtained Logistic-Normal distribution with Dirichlet one. 

| Dirichlet (with random parameters) | Logistic-Normal (approximation to Dirichlet) | Dirichlet (approximation to obtained Logistic-Normal) |
| :--: | :--: | :--: |
| ![Closed-form Laplace Bridge demonstration](laplace-bridge-1.png) | ![Closed-form Laplace Bridge demonstration](laplace-bridge-2.png) | ![Closed-form Laplace Bridge demonstration](laplace-bridge-3.png) |

**REINFORCE in Acrobot environment.** In this part we train an Agent in the [Acrobot environment](https://www.gymlibrary.dev/environments/classic_control/acrobot/), using REINFORCE to make optimization steps.

**VAE with discrete latents.** All the other 6 algorithms are used to train a VAE with discrete latents. Each of the discussed relaxation techniques allows us to learn the latent space with the corresponding distribution. All implemented distributions have a similar structure, so we chose one distribution for demonstration and conducted a number of experiments with it. **Correlated Relaxed Bernoulli** was chosen as a demonstration method. This method generates correlated gate vectors from a multivariate Bernoulli distribution using a Gaussian copula. We define the parameters $\pi$, $R$, and $\tau$ as follows:

- Tensor $\pi$, representing the probabilities of the Bernoulli distribution, with an event shape of 3 and a batch size of 2:

$$
\pi = \begin{bmatrix}
0.2 & 0.4 & 0.4 \\
0.3 & 0.5 & 0.2
\end{bmatrix}
$$

- Correlation matrix $R$ for the Gaussian copula:

$$
R = \begin{bmatrix}
1.0 & 0.5 & 0.3 \\
0.5 & 1.0 & 0.7 \\
0.3 & 0.7 & 1.0
\end{bmatrix}
$$

- Temperature hyperparameter $\tau = 0.1$

Finally, after training we obtained reconstruction and sampling results for a MNIST dataset that we provide below. We see that VAE has learned something adequate, which means that the reparameterization is happening correctly. For the rest of the methods, VAE are also implemented, which you can get engaged using scripts in the demo experiments directory.

{{< figure src="correlated_bernoulli_reconstruction.png" title="Fig. 2. Variational Autoencoder (VAE) with discrete Correlated Relaxed Bernoulli latents. Reconstruction." >}}
{{< figure src="correlated_bernoulli_sample.png" title="Fig. 3. Variational Autoencoder (VAE) with discrete Correlated Relaxed Bernoulli latents. Sampling." >}}

## Conclusion

In summary, ``Just Relax It`` is a powerful tool for researchers and practitioners working with discrete variables in neural networks. By offering a comprehensive set of relaxation techniques, our library aims to make the optimization process more efficient and accessible. We encourage you to explore our library, try out the demo, and contribute to its development. Together, we can push the boundaries of what is possible with discrete variable relaxation in machine learning.

Thank you for reading, and happy coding!

[Daniil Dorin](https://github.com/DorinDaniil), [Igor Ignashin](https://github.com/ThunderstormXX), [**Nikita Kiselev**](https://kisnikser.github.io/), [Andrey Veprikov](https://github.com/Vepricov)

## References
