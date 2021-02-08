# [Course 2 Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning)

## 1. Practical Aspects of Deep Learning

### 1.1. Setting Up A Machine Learning Application

#### 1.1.1. Train / Dev / Test Sets

Applied ML is a highly **iterative** process.

**Make sure that Dev and Test sets come from the same distribution.**



#### 1.1.2. Bias and Variance

* **High Bias** (depends on Train set performance): underfitting

* **High Variance** (depends on Dev set performance): overfitting

**Train set error vs. Dev set error** comparing to **optimal error** (sometimes called **base error** or **Bayes error**)



#### 1.1.3. Basic Recipe for Machine Learning

* **High Bias**: bigger NN, train longer (and NN architectures search)

* **High Variance**: **more data**, regularization (and NN architectures search)

**Training a bigger NN almost never hurts so long as we have a well regularized network. And the main cost of training such a NN is just computational time.**



### 1.2. Regularizing Neural Network

#### 1.2.1. Regularization

**Logistic Regression**

* **L2 Regularization**
  $$
  J(\mathbf{w}, b) = \frac{1}{m_{\text{train}}}\sum_{i=1}^{m_{\text{train}}} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m_{\text{train}}}\|\mathbf{w}\|_{2}^{2}
  $$
  where $$\lambda$$ is the regularization parameter. Almost all the parameters are in $$\mathbf{w}$$ rather $$b$$, so $$b$$ is usually not included in the regularization term. **L2 regularization is the most common type of regularization.**

* **L1 Regularization**
  $$
  J(\mathbf{w}, b) = \frac{1}{m_{\text{train}}}\sum_{i=1}^{m_{\text{train}}} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{m_{\text{train}}}\|\mathbf{w}\|_{1}
  $$
  If we use L1 regularization, then $$\mathbf{w}$$ will end up being sparse, i.e., $$\mathbf{w}$$ will have a lot of zeros in it. In practice, it helps only a little bit.



**Neural Network L2 Regularization**
$$
J(\mathbf{W}^{[1]}, \mathbf{b}^{[1]}, \cdots , \mathbf{W}^{[L]}, \mathbf{b}^{[L]}) = \frac{1}{m_{\text{train}}}\sum_{i=1}^{m_{\text{train}}} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m_{\text{train}}}\sum_{l=1}^{L} \|\mathbf{W}^{[l]}\|_{F}^{2}
$$
**Backpropagation**
$$
d\mathbf{W}^{[l]} = d\mathbf{W}^{[l]}_{\text{origin}} + \frac{\lambda}{m_{\text{train}}} \mathbf{W}^{[l]}
$$
where $$d\mathbf{W}^{[l]}_{\text{origin}}$$ is the original gradient without regularization.
$$
\begin{align}
\mathbf{W}^{[l]} &= \mathbf{W}^{[l]} - d\mathbf{W}^{[l]} \\
&= (1-\frac{\lambda}{m_{\text{train}}}) \cdot \mathbf{W}^{[l]} - d\mathbf{W}^{[l]}_{\text{origin}}
\end{align}
$$
where $$1-\frac{\lambda}{m_{\text{train}}} < 1$$, so L2 regularization is sometimes also called weight decay.



Assuming $$\mathbf{x} \in \mathbb{R}^{n}$$,

**Taxicab norm or Manhattan norm, $$l_{1}$$ norm**
$$
\|\mathbf{x}\|_{1} = \sum_{i=1}^{n} |x_{i}|
$$
**Euclidean norm or $$l_{2}$$ norm**
$$
\|\mathbf{x}\|_{2} = \sqrt{\sum_{i=1}^{n} |x_{i}|^{2}}
$$
**p-norm or $$l_{p}$$ norm**
$$
\|\mathbf{x}\|_{p} = (\sum_{i=1}^{n} |x_{i}|^{p})^{1/p}
$$
where $$p \geq 1$$.



Assuming $$\mathbf{A} \in \mathbb{R}^{m \times n}$$,

**Frobenius norm**
$$
\|\mathbf{A}\|_{F} = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^{2}}
$$

#### 1.2.2. Why Regularization Reduces Overfitting?

Extra regularization term is added to penalize the weights (or Frobenius norm) from being too large. **The intuition is** that by cranking up $$\lambda$$ to be really big, the weights will be set close to zero for a lot of hidden units (**zeroing out the impact of hidden units**). And if that's the case, then this much simplified neural network becomes a much smaller neural network. In practice, this actually won't happen.

By tuning the hyperparameter $$\lambda$$, it turns out that what actually happens is NN would still use all the hidden units, but each of them would just have a much smaller effect. Finally, we do end up with a simpler network and as if we have a smaller network that is therefore less prone to overfitting.

#### 1.2.3. Dropout Regularization

**Implementing dropout: inverted dropout**
$$
\mathbf{a}^{[l](i)} = \mathbf{m}^{[l]} \odot \mathbf{a}^{[l](i)}
$$
where $$\odot$$ is element-wise multiplication, $$\mathbf{m}^{[l]} \in \mathbb{R}^{n_{l}}$$ is a binary mask to directly eliminate the impact of a certain hidden units with some probability $$1 - \text{keep_prob}$$.
$$
\mathbf{a}^{[l](i)} = \frac{1}{\text{keep_prob}} \cdot \mathbf{a}^{[l](i)}
$$
which ensures that the expected value of $$\mathbf{a}^{[l](i)}$$ remains the same.

**Making predictions at test time**

**We don't use dropout or add an extra scaling parameter at test time.**

#### 1.2.4. Why Does Dropout Work?

Each iteration we're training with a smaller neural network, so it should have a regularization effect.

With drop out, all the features can get randomly eliminated, then the weights are spread out, this will tend to have an effect of shrinking Frobenius norm of the weights. So, similar to L2 regularization, it penalize the weights from being too large which helps prevent overfitting.

**One big downside of drop out is that the cost function is no longer well-defined.**

#### 1.2.5. Other Regularization

* **Data Augmentation**

* **Early Stopping**

  By stopping gradient decent early, we're sort of mixing optimizing cost function and trying to not overfit together. But these two goals are kind of orthogonal. The advantage of it is that running the gradient descent process just once.



### 1.3. Setting Up Optimization Problem

#### 1.3.1. Normalizing Inputs

$$
\begin{align}
\mu &= \frac{1}{m} \sum_{i}^{m} \mathbf{x}^{(i)} \\
\sigma^{2} &= \frac{1}{m} \sum_{i}^{m} (\mathbf{x}^{(i)} - \mu)^{2} \\
\end{align}
$$

where the square is the element-wise operation, $$\mu, \sigma^{2} \in \mathbb{R}^{n_{x}}$$.
$$
\mathbf{x}^{(i)}_{\text{norm}} = \frac{\mathbf{x}^{(i)} - \mu}{\sigma + \epsilon}
$$
where the divide is the element-wise operation, we usually choose $$\epsilon = 10^{-8}$$.

#### 1.3.2. Why Normalizing Inputs

Normalizing inputs guarantees that all the features take on a similar scale, which can speed up learning. And performing this type of normalization never does any harm.

#### 1.3.3. Vanishing and Exploding Gradients

With a deep neural network, activations or gradients increase or decrease exponentially as a function of $$L$$, then these values could get really big (exploding) or really small (vanishing).

This makes learning difficult, especially if gradients are getting exponentially smaller, then gradient descent will take tiny little steps, and it will take a long time for gradient descent to learn anything.

It turns out there is a partial solution that is **careful choice of initializing the weights**.

#### 1.3.4. Weight Initialization for Deep Neural Networks

For $$\mathbf{W}^{[l]}$$,

* when using `tanh`, initialize it so that the variance of the weights is $$\frac{1}{n_{l-1}}$$, or Xavier initialization, initialize it so that the variance of the weights is $$\frac{2}{n_{l} + n_{l-1}}$$;
* when using `ReLU`, initialize it so that the variance of the weights is $$\frac{2}{n_{l-1}}$$.

#### 1.3.5. Numerical Approximation of Gradients

$$
f^{\prime}(\theta) = \lim_{\epsilon \rightarrow 0}\frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}
$$

#### 1.3.6. Gradient Checking

Estimate $$d\theta^{[i]\{t\}}$$,
$$
d\theta_{\text{approx}}^{[i]\{t\}} = \frac{J(\theta^{[1]\{t\}}, \theta^{[2]\{t\}}, \cdots , \theta^{[i]\{t\}} + \epsilon, \cdots) - J(\theta^{[1]\{t\}}, \theta^{[2]\{t\}}, \cdots , \theta^{[i]\{t\}} - \epsilon, \cdots)}{2\epsilon}
$$
Check if $$d\theta_{\text{approx}}^{[i]\{t\}} \approx d\theta^{[i]\{t\}}$$, i.e.
$$
\frac{\| d\theta_{\text{approx}}^{[i]\{t\}} - d\theta^{[i]\{t\}}\|_{2}}{\|d\theta_{\text{approx}}^{[i]\{t\}}\|_{2} + \|d\theta^{[i]\{t\}}\|_{2}} \leq 10^{-7}
$$

#### 1.3.7. Gradient Checking Implementation Notes

Remember regularization term, but also notice that it doesn’t work with dropout.



## 2. Optimization Algorithms

### 2.1. Min-batch Gradient Descent

#### 2.1.1. Batch vs. Min-Batch Gradient Descent

* Batch
  $$
  (\mathbf{X}, \mathbf{Y})
  $$

* Mini-batch
  $$
  \{(\mathbf{X}^{\{1\}}, \mathbf{Y}^{\{1\}}), (\mathbf{X}^{\{2\}}, \mathbf{Y}^{\{2\}}), \cdots, (\mathbf{X}^{\{T\}}, \mathbf{Y}^{\{T\}})\}
  $$

When we have a large training set, mini-batch gradient descent runs much faster than batch gradient descent (**speed up the learning**). With mini-batch gradient descent, a single pass through Train set, i.e., one epoch, allows us to take $$T$$ gradient descent steps.

#### 2.1.2. Understanding Min-Batch Gradient Descents

Choosing mini-batch size $$m \in [1, m_{\text{train}}]$$,

* $$m=m_{\text{train}}$$: batch gradient descent

* $$m=1$$: stochastic gradient descent

Shuffle

Partition



### 2.2. Exponentially Weighted Averages

A moving average (MA) is commonly used with time-series data to smooth out short-term fluctuations and highlight longer-term trends or cycles.

#### 2.2.1. Exponential Moving Average (EMA)

$$
v_{t}=\left\{\begin{array}{ll}
y_{1}, & t=1 \\
\beta \cdot v_{t-1} + (1 - \beta) \cdot y_{t}, & t>1
\end{array}\right.
$$

where $$y_{t}$$ is the observation at $$t$$, $$v_{t}$$ is the value of the EMA at $$t$$, smoothing factor $$\beta \in (0, 1)$$,  which represents the degree of weighting decrease, a lower $$\beta$$ discounts older observations faster. $$v_{t}$$ is the exponentially weighted average over the time period $$\frac{1}{1 - \beta}$$.

#### 2.2.2. Understanding Exponentially Weighted Averages

Weights decays exponentially
$$
\begin{align}
v_{t}&=\beta \cdot v_{t-1} + (1 - \beta) \cdot y_{t} \\
&=(1 - \beta) \cdot y_{t} +  \beta(1 - \beta) \cdot y_{t-1} + \cdots + \beta^{t-1}(1 - \beta) \cdot y_{1}
\end{align}
$$

#### 2.2.3. Bias Correction in Exponentially Weighted Averages

$$
v_{t} = \frac{v_{t}}{1 - \beta^{t}}
$$

where when $$t$$ is small, $$1 - \beta^{t} < 1$$, when $$t$$ is large, $$1 - \beta^{t} \approx 1$$.



### 2.3. Gradient Descent with Momentum

Momentum
$$
\begin{align}
\mathbf{V}^{[l]}_{d\mathbf{W}^{[l]}} &= \beta \cdot \mathbf{V}^{[l]}_{d\mathbf{W}^{[l]}} + (1 - \beta) \cdot d\mathbf{W}^{[l]} \\
\mathbf{V}^{[l]}_{d\mathbf{b}^{[l]}} &= \beta \cdot \mathbf{V}^{[l]}_{d\mathbf{b}^{[l]}} + (1 - \beta) \cdot d\mathbf{b}^{[l]}
\end{align}
$$
where initially, $$\mathbf{V}^{[l]}_{d\mathbf{W}^{[l]}} = 0$$, $$\mathbf{V}^{[l]}_{d\mathbf{b}^{[l]}} = 0$$, we usually choose $$\beta = 0.9$$, i.e., $$\mathbf{V}^{[l]}_{d\mathbf{W}^{[l]}}$$ and $$\mathbf{V}^{[l]}_{d\mathbf{b}^{[l]}}$$ are the exponentially wighted averages over 10 gradients.
$$
\begin{align}
\mathbf{W}^{[l]} &= \mathbf{W}^{[l]} - \alpha \cdot \mathbf{V}^{[l]}_{d\mathbf{W}^{[l]}} \\
\mathbf{b}^{[l]} &= \mathbf{b}^{[l]} - \alpha \cdot \mathbf{V}^{[l]}_{d\mathbf{b}^{[l]}}
\end{align}
$$


### 2.4. RMSprop

Root Mean Square Prop (RMSprop)
$$
\begin{align}
\mathbf{S}^{[l]}_{d\mathbf{W}^{[l]}} &= \beta \cdot \mathbf{S}^{[l]}_{d\mathbf{W}^{[l]}} + (1 - \beta) \cdot (d\mathbf{W}^{[l]})^{2} \\
\mathbf{S}^{[l]}_{d\mathbf{b}^{[l]}} &= \beta \cdot \mathbf{S}^{[l]}_{d\mathbf{b}^{[l]}} + (1 - \beta) \cdot (d\mathbf{b}^{[l]})^{2}
\end{align}
$$
where the square is an element-wise operation, initially, $$\mathbf{S}^{[l]}_{d\mathbf{W}^{[l]}} = 0$$, $$\mathbf{S}^{[l]}_{d\mathbf{b}^{[l]}} = 0$$, and we usually choose $$\beta = 0.999$$.
$$
\begin{align}
\mathbf{W}^{[l]} &= \mathbf{W}^{[l]} - \alpha \cdot \frac{d\mathbf{W}^{[l]}}{\sqrt{\mathbf{S}^{[l]}_{d\mathbf{W}^{[l]}}} + \epsilon} \\
\mathbf{b}^{[l]} &= \mathbf{b}^{[l]} - \alpha \cdot \frac{d\mathbf{b}^{[l]}}{\sqrt{\mathbf{S}^{[l]}_{d\mathbf{b}^{[l]}}} + \epsilon}
\end{align}
$$
where we usually choose $$\epsilon = 10^{-8}$$.

RMSprop, similar to momentum, has the effects of damping out the oscillations in mini-batch gradient descent.



### 2.5. Adam Optimization Algorithm

Adaptive Moment Estimation (Adam)
$$
\mathbf{V}^{[l]}_{d\mathbf{W}^{[l]}} = \beta_{1} \cdot \mathbf{V}^{[l]}_{d\mathbf{W}^{[l]}} + (1 - \beta_{1}) \cdot d\mathbf{W}^{[l]} \text{, }
\mathbf{V}^{[l]}_{d\mathbf{b}^{[l]}} = \beta_{1} \cdot \mathbf{V}^{[l]}_{d\mathbf{b}^{[l]}} + (1 - \beta_{1}) \cdot d\mathbf{b}^{[l]} \\
\mathbf{S}^{[l]}_{d\mathbf{W}^{[l]}} = \beta_{2} \cdot \mathbf{S}^{[l]}_{d\mathbf{W}^{[l]}} + (1 - \beta_{2}) \cdot (d\mathbf{W}^{[l]})^{2} \text{, }
\mathbf{S}^{[l]}_{d\mathbf{b}^{[l]}} = \beta_{2} \cdot \mathbf{S}^{[l]}_{d\mathbf{b}^{[l]}} + (1 - \beta_{2}) \cdot (d\mathbf{b}^{[l]})^{2}
$$
where initially, $$\mathbf{V}^{[l]}_{d\mathbf{W}^{[l]}} = 0$$, $$\mathbf{V}^{[l]}_{d\mathbf{b}^{[l]}} = 0$$, $$\mathbf{S}^{[l]}_{d\mathbf{W}^{[l]}} = 0$$, $$\mathbf{S}^{[l]}_{d\mathbf{b}^{[l]}} = 0$$, and we usually choose $$\beta_{1} = 0.9$$, $$\beta_{2} = 0.999$$.
$$
\mathbf{V}^{[l] \text{corrected}}_{d\mathbf{W}^{[l]}} = \frac{\mathbf{V}^{[l]}_{d\mathbf{W}^{[l]}}}{1 - \beta_{1}^{t}} \text{, }
\mathbf{V}^{[l] \text{corrected}}_{d\mathbf{b}^{[l]}} = \frac{\mathbf{V}^{[l]}_{d\mathbf{b}^{[l]}}}{1 - \beta_{1}^{t}} \\
\mathbf{S}^{[l] \text{corrected}}_{d\mathbf{W}^{[l]}} = \frac{\mathbf{S}^{[l]}_{d\mathbf{W}^{[l]}}}{1 - \beta_{2}^{t}} \text{, }
\mathbf{S}^{[l] \text{corrected}}_{d\mathbf{b}^{[l]}} = \frac{\mathbf{S}^{[l]}_{d\mathbf{b}^{[l]}}}{1 - \beta_{2}^{t}}
$$
where $t$ is the iterations or mini-batch index.
$$
\begin{align}
\mathbf{W}^{[l]} &= \mathbf{W}^{[l]} - \alpha \cdot \frac{\mathbf{V}^{[l] \text{corrected}}_{d\mathbf{W}^{[l]}}}{\sqrt{\mathbf{S}^{[l] \text{corrected}}_{d\mathbf{W}^{[l]}}} + \epsilon} \\
\mathbf{b}^{[l]} &= \mathbf{b}^{[l]} - \alpha \cdot \frac{\mathbf{V}^{[l] \text{corrected}}_{d\mathbf{b}^{[l]}}}{\sqrt{\mathbf{S}^{[l] \text{corrected}}_{d\mathbf{b}^{[l]}}} + \epsilon}
\end{align}
$$
where we usually choose $$\epsilon = 10^{-8}$$.



### 2.6. Learning Rate Decay

Learning rate decay methods

* $$
  \alpha = \frac{1}{1 + \text{decay_rate} \cdot \text{epoch}} \alpha_{0}
  $$



* Exponential decay
  $$
  \alpha = 0.95^{\text{epoch}} \alpha_{0}
  $$




* $$
  \alpha = \frac{k}{\sqrt{\text{epoch}}} \alpha_{0} \text{ or } \alpha = \frac{k}{\sqrt{t}} \alpha_{0}
  $$



* Use a learning rate that decreases in discrete steps
* Manual decay



### 2.7. The Problem of Local Optima

We're much more likely to see saddle points than local optimum.

It turns out that plateaus can really slow down learning and a plateau is a region where the derivative is close to zero for a long time. Adam can actually speed up the rate at which we move down the plateau and then get off the plateau.



## 3. Hyperparameter Tuning, Batch Normalization and Programming Frameworks

### 3.1. Hyperparameter Tuning

#### 3.1.1. Tuning Process

Use random sampling (picking hyperparameters at random) and adequate search and optionally consider implementing a coarse to fine search process.

#### 3.1.2. Using An Appropriate Scale to Pick Hyperparameters

Sample uniformly at random using an appropriate scale.

#### 3.1.3. Hyperparameter Tuning in Practical: Pandas vs. Caviar

* Panda approach: babysitting one model
* Caviar approach: training many models in parallel



### 3.2. Batch Normalization

#### 3.2.1. Normalizing Activations in A Neural Network

Batch nomalizing $$\mathbf{z}^{[l](i)}$$ instead of $$\mathbf{a}^{[l](i)}$$,
$$
\begin{align}
\mu^{[l]} &= \frac{1}{m} \sum_{i}^{m} \mathbf{z}^{[l](i)} \\
(\sigma^{[l]})^{2} &= \frac{1}{m} \sum_{i}^{m} (\mathbf{z}^{[l](i)} - \mu^{[l]})^{2} \\
\end{align}
$$
where the square is the element-wise operation, $$\mu^{[l]}, (\sigma^{[l]})^{2} \in \mathbb{R}^{n_{l}}$$.
$$
\begin{align}
\mathbf{z}^{[l](i)}_{\text{norm}} &= \frac{\mathbf{z}^{[l](i)} - \mu^{[l]}}{\sqrt{(\sigma^{[l]})^{2}} + \epsilon} \\
\tilde{\mathbf{z}}^{[l](i)} &= \gamma^{[l]} \odot \mathbf{z}^{[l](i)}_{\text{norm}} + \beta^{[l]} \\
\end{align}
$$
where the divide, $$\odot$$ are the element-wise operations, we usually choose $$\epsilon = 10^{-8}$$, $$\gamma^{[l]}, \beta^{[l]} \in \mathbb{R}^{n_{l}}$$, if $$\gamma^{[l]} = \sqrt{(\sigma^{[l]})^{2}} + \epsilon$$ and $$\beta^{[l]} = \mu^{[l]}$$, then $$\tilde{\mathbf{z}}^{[l](i)} =  \mathbf{z}^{[l](i)}$$.
$$
\begin{align}
\mathbf{a}^{[l](i)} &= g^{[l]}(\tilde{\mathbf{z}}^{[l](i)}) \\
\mathbf{z}^{[l+1](i)} &= \mathbf{W}^{[l+1]} \mathbf{a}^{[l](i)}\\
\end{align}
$$
where $$\beta^{[l]}$$ has the effect of the shift or the bias.

#### 3.2.2. Fitting Batch Norm into A Neural Network

$$
\begin{align}
\mathbf{W}^{[l]} &= \mathbf{W}^{[l]} - \alpha \cdot d\mathbf{W}^{[l]} \\
\gamma^{[l]} &= \gamma^{[l]} - \alpha \cdot d\gamma^{[l]} \\
\beta^{[l]} &= \beta^{[l]}  - \alpha \cdot d\beta^{[l]}
\end{align}
$$

#### 3.2.3. Why Does Batch Norm Work?

**Speeding up learning**

By normalizing all the features to take on a similar range of values, **batch norm can speed up learning**.

**Learning on shifting input distribution**

Batch norm ensures that no matter how the earlier layers' parameters change, the mean and variance of each later layers' unit will remain the same (stable), not necessarily mean zero and variance one, but whatever value is governed by $$\beta^{[l]}$$ and $$\gamma^{[l]}$$.

So that **the later layers of the neural network has more firm ground to stand on**. And even though the input distribution changes a bit, even as the earlier layers keep learning, the amount that it forces the later layers to adapt to is reduced. So, **batch norm mitigates the problem of covariate shift.**  

Batch norm weakens the coupling between what the early layers parameters has to do and what the later layers parameters have to do. So it allows each layer of the network to learn by itself, a little bit more independently of other layers, and this has the effect of **speeding up of learning** in the whole network.

**Batch norm as regularization**

Each mini-batch is scaled by the mean and variance computed on just that one mini-batch, which adds some noise to each hidden layer's activations. So, similar to dropout, **batch norm therefore has a slight regularization effect**.

Because the noise added is quite small, this is not a huge regularization effect, we might use batch norm together with dropouts if we want the more powerful regularization effect of dropout.

By using a larger mini-batch size, we're reducing this noise and therefore also reducing this regularization effect. So, really, **don't turn to batch norm as a regularization**.

#### 3.2.4. Batch Norm at Test Time

At test time, we need to process a single example at a time, in practice, we could estimate $$\mu^{[l]\{T\}}$$ and $$(\sigma^{[l]\{T\}})^{2}$$ using a exponentially weighted average over Train set.



### 3.3. Multi-Class Classification

#### 3.3.1. Softmax Regression

Assuming we have $$c$$ classes,
$$
\hat{\mathbf{y}}^{(i)} = \mathbf{a}^{[L](i)} = \operatorname{Softmax}(\mathbf{z}^{[L](i)})
$$
where $$\hat{\mathbf{y}}^{(i)}, \mathbf{a}^{[L](i)}, \mathbf{z}^{[L](i)} \in \mathbb{R}^{c}$$.
$$
a^{[L](i)}_{j} = \frac{e^{z^{[L](i)}_{j}}}{\sum_{j=1}^{c} e^{z^{[L](i)}_{j}}}
$$
and
$$
\sum_{j=1}^{c} a^{[L](i)}_{j} = 1
$$

#### 3.3.2. Training A Softmax Classifier

Loss function
$$
\mathcal{L}(\hat{\mathbf{y}}^{(i)}, \mathbf{y}^{(i)}) = - \sum_{j=1}^{c} y^{(i)}_{j}\log \hat{y}^{(i)}_{j}
$$
where $$\mathbf{y}^{(i)}$$ is one-hot column vector and $$\mathbf{y}^{(i)} \in \mathbb{R}^{c}$$.

Backpropagation
$$
d\mathbf{z}^{[L](i)} = \mathbf{a}^{[L](i)} - \mathbf{y}^{(i)}
$$


### 3.4. Introduction to Programming Frameworks

#### 3.4.1. Deep Learning Frameworks

Criteria in choosing a deep learning framework:

* ease of programming (including both development and deployment)

* running speed

* whether or not the framework is truly open (open source with good governance)

#### 3.4.2. TensorFlow
