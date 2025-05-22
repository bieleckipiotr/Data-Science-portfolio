### 🧮 REBAR_Optimizer  
REBAR_Optimizer is a Python implementation and mathematical exploration of the REBAR gradient estimator, designed for optimizing expectations over binary(or multiclass) random variables.

The REBAR estimator is a variance-reduced, unbiased gradient estimator that combines:

The REINFORCE score-function method, a continuous relaxation (Gumbel-softmax), the reparameterization trick, and a control variate correction term.  

📘 Project Objective:  
The objective of this project is twofold:  

- Implement and derive the REBAR estimator from the original paper.  

- Compare its performance against the traditional REINFORCE estimator and the Expectation-Maximization (EM) algorithm, on a benchmark task - the Gaussian Mixture Model (GMM).

🧠 Mathematical Context:  
We are interested in optimizing the following expectation:  
$\mathbb{E}_{b \sim p(b|\theta)}[f(b, \theta)]$  
where:  
- $b \in \{0, 1\}^K$ is a vector of independent binary random variables,  

- $\theta \in \mathbb{R}^K$ parametrizes their Bernoulli distributions,  

- $f(b, \theta)$ is a differentiable function of both $b$ and $\theta$.  

The challenge:  

The expectation is non-differentiable with respect to $\theta$, due to the discrete nature of $b$.  

🔁 Traditional Method:  
The REINFORCE estimator approximates the gradient as:

$\nabla_\theta \mathbb{E}_{b}[f(b, \theta)] \approx f(b, \theta) \nabla_\theta \log p(b|\theta)$  
This estimator is unbiased, but often suffers from high variance.

🔧 REBAR Estimator:  
REBAR improves on this by introducing:

A continuous relaxation $\tilde{b} \sim q(\tilde{b}|\theta)$,  

A reparameterized sample $z \sim p(z|\theta)$,  

A control variate based on $f(\tilde{b}, \theta)$,  

to reduce variance while keeping the gradient estimate unbiased.  

The continuous relaxation for binary variables typically uses the Binary Concrete distribution:  
$$ \tilde{b} = \sigma\left( \frac{1}{\lambda} \left( \log \theta - \log (1 - \theta) + \log u - \log (1 - u) \right) \right), \quad u \sim \text{Uniform}(0,1) $$
This uses the sigmoid function $\sigma(\cdot)$ as a soft approximation of discrete sampling.

📊 Experimental Setup:  
Task  
Gaussian Mixture Model (GMM):

Binary latent variables represent component selection.  

Objective is to maximize ELBO or marginal log-likelihood.  

Estimators Compared:  
REINFORCE (score-function estimator)  

REBAR - proves better than REINFORCE.  

Expectation-Maximization (EM) baseline - hard to beat, as this is the go-to approach for Gaussian Mixture Models.
