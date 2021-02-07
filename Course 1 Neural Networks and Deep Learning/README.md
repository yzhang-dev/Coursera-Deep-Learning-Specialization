# [Course 1 Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning)

## 1. Introduction to Deep Learning

### 1.1. What Is A Neural Network?

### 1.2. Supervised Learning with Neural Networks

### 1.3. Why Is Deep Learning Taking Off?

Scale drives deep learning progress.



## 2. Basics of Neural Networks

### 2.1. Binary Classification

Training set with $$m_{\text{train}}$$ examples:
$$
\{(\mathbf{x}^{(1)}, y^{(1)}), (\mathbf{x}^{(2)}, y^{(2)}), \dots , (\mathbf{x}^{(m_{\text{train}})}, y^{(m_{\text{train}})})\}
$$
where $$\mathbf{x}^{(i)} \in \mathbb{R}^{n_{x}}$$, $$y^{(i)} \in \{0, 1\}$$, $$1 \leq i \leq m_{\text{train}}$$.

Or
$$
(\mathbf{X}, \mathbf{Y})
$$
where $$\mathbf{X} = [\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots , \mathbf{x}^{(m_{\text{train}})}]$$, $$\mathbf{X} \in \mathbb{R}^{n_{x} \times m_{\text{train}}}$$, $$\mathbf{Y} = [y^{(1)}, y^{(2)}, \dots , y^{(m_{\text{train}})}]$$, $$\mathbf{Y} \in \mathbb{R}^{1 \times m_{\text{train}}}$$.



### 2.2. Logistic Regression

Given single example $$(\mathbf{x}^{(i)}, y^{(i)})$$, where $$\mathbf{x}^{(i)} \in \mathbb{R}^{n_{x}}$$, $$y^{(i)} \in \{0, 1\}$$, $$1 \leq i \leq m_{\text{train}}$$, the prediction $$\hat{y}^{(i)} = P(y^{(i)}=1|\mathbf{x}^{(i)})$$:
$$
\begin{align}
z^{(i)} &= \mathbf{w}^{\top} \mathbf{x}^{(i)} + b \\
\hat{y}^{(i)} &= a^{(i)} = \operatorname{Sigmoid}(z^{(i)}) = \sigma(z^{(i)})
\end{align}
$$
where the parameters $$\mathbf{w} \in \mathbb{R}^{n_{x}}$$, $$b \in \mathbb{R}$$, and the activation function $$\operatorname{Sigmoid}(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$$.

Given $$m_{\text{train}}$$ examples, the predictions $$\hat{\mathbf{Y}} = [\hat{y}^{(i)}, \hat{y}^{(i)}, \dots , \hat{y}^{(m_{\text{train}})}]$$:
$$
\begin{align}
\mathbf{Z} &= \mathbf{w}^{\top} \mathbf{X} + b \\
\hat{\mathbf{Y}} &= \mathbf{A} = \operatorname{Sigmoid}(\mathbf{Z}) = \sigma(\mathbf{Z})
\end{align}
$$



### 2.3. Logistic Regression Cost Function

Given the training set with $$m_{\text{train}}$$ examples: $$\{(\mathbf{x}^{(1)}, y^{(1)}), (\mathbf{x}^{(2)}, y^{(2)}), \dots , (\mathbf{x}^{(m_{\text{train}})}, y^{(m_{\text{train}})})\}$$, we expect $$\hat{y}^{(i)} \approx y^{(i)}$$, $$1 \leq i \leq m_{\text{train}}$$, i.e., we want to find the parameters $$\mathbf{w}$$, $$b$$ that minimize the loss function $$J(\mathbf{w}, b)$$.

Loss function:
$$
\begin{align}
\mathcal{L}(\hat{y}^{(i)}, y^{(i)}) &= -(y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)})) \\
J(\mathbf{w}, b) &= \frac{1}{m_{\text{train}}}\sum_{i=1}^{m_{\text{train}}} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})
\end{align}
$$

> Why not mean squared error (MSE) or one half MSE?
>
> In logistic regression people don't usually use MSE because when we come to learn the parameters, we'll find that the optimization problem becomes non-convex. Gradient descent may not find the global optimum. So we'll end up with optimization problem with multiple local optima.
>
> **We need to define a loss function that will give us an optimization problem that is convex.**



### 2.4. Explanation of Logistic Regression Cost Function

Given single example $$(\mathbf{x}^{(i)}, y^{(i)})$$, where $$\mathbf{x}^{(i)} \in \mathbb{R}^{n_{x}}$$, $$y^{(i)} \in \{0, 1\}$$, $$1 \leq i \leq m_{\text{train}}$$, the prediction $$\hat{y}^{(i)} = P(y^{(i)}=1|\mathbf{x}^{(i)})$$. So we have
$$
\begin{align}
&P(y^{(i)}=1|\mathbf{x}^{(i)}) = \hat{y}^{(i)} \\
&P(y^{(i)}=0|\mathbf{x}^{(i)}) = 1 - \hat{y}^{(i)}
\end{align}
$$
Or
$$
\begin{align}
P(y^{(i)}|\mathbf{x}^{(i)}) &= (\hat{y}^{(i)})^{y^{(i)}} \cdot (1 - \hat{y}^{(i)})^{1 - y^{(i)}} \\
\log P(y^{(i)}|\mathbf{x}^{(i)}) &= y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)})
\end{align}
$$
**So minimizing the loss corresponds to maximizing the log of the probability.**

Given the training set with $$m_{\text{train}}$$ examples: $$\{(\mathbf{x}^{(1)}, y^{(1)}), (\mathbf{x}^{(2)}, y^{(2)}), \dots , (\mathbf{x}^{(m_{\text{train}})}, y^{(m_{\text{train}})})\}$$, let's just assume that the training examples are identically independently distributed (IID), then
$$
\begin{align}
P(\mathbf{Y}|\mathbf{X}) &= \prod_{i}^{m_{\text{train}}}P(y^{(i)}|\mathbf{x}^{(i)}) \\
\log P(\mathbf{Y}|\mathbf{X}) &= \sum_{i}^{m_{\text{train}}} \log P(y^{(i)}|\mathbf{x}^{(i)})
\end{align}
$$
**So, by minimizing this cost function, we're actually carrying out maximum likelihood estimation (MLE) with the logistic regression model.**

>To make sure that our quantities are better scaled, we just add a 1 over $$m_{\text{train}}$$ extra scaling factor there.



### 2.5. Logistic Regression Gradient Descent

We want to find the parameters $$\mathbf{w}$$, $$b$$ that minimize the loss function $$J(\mathbf{w}, b)$$.

Gradient descent:
$$
\begin{align}
w_{j} &= w_{j} - \alpha \cdot \frac{\partial J(\mathbf{w}, b)}{\partial w_{j}} = w_{j} - \alpha \cdot dw_{j} \text{, } 1 \leq j \leq n_{x} \\
b &= b - \alpha \cdot \frac{\partial J(\mathbf{w}, b)}{\partial b} = b - \alpha \cdot db
\end{align}
$$
where $$\mathbf{w}^{\top} = [w_{1}, w_{2}, \dots, w_{n_{x}}]$$.

Given single example $$(\mathbf{x}^{(i)}, y^{(i)})$$, where $$\mathbf{x}^{(i)} \in \mathbb{R}^{n_{x}}$$, $$y^{(i)} \in \{0, 1\}$$, $$1 \leq i \leq m_{\text{train}}$$,
$$
\begin{align}
&da^{(i)} = \frac{\partial \mathcal{L}}{\partial \hat{y}^{(i)}} = -\frac{y^{(i)}}{a^{(i)}} + \frac{1 - y^{(i)}}{1 - a^{(i)}}\\
&dz^{(i)} = \frac{\partial \mathcal{L}}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} = a^{(i)}(1−a^{(i)}) \cdot da^{(i)} = a^{(i)} - y^{(i)}\\
&dw_{j} = \frac{\partial J(\mathbf{w}, b)}{\partial w_{j}} = \frac{\partial \mathcal{L}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial w_{j}} = x^{(i)}_{j} \cdot dz^{(i)} \\
&db = \frac{\partial J(\mathbf{w}, b)}{\partial b} = \frac{\partial \mathcal{L}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial b} = dz^{(i)}\\
\end{align}
$$



### 2.6. Gradient Descent on m Examples

Given the training set with $$m_{\text{train}}$$ examples: $$\{(\mathbf{x}^{(1)}, y^{(1)}), (\mathbf{x}^{(2)}, y^{(2)}), \dots , (\mathbf{x}^{(m_{\text{train}})}, y^{(m_{\text{train}})})\}$$,
$$
\begin{align}
&dz^{(i)} = a^{(i)} - y^{(i)}\\
&dw_{j} = \frac{\partial J(\mathbf{w}, b)}{\partial w_{j}} = \frac{1}{m_{\text{train}}}\sum_{i=1}^{m_{\text{train}}} \frac{\partial \mathcal{L}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial w_{j}} = \frac{1}{m_{\text{train}}}\sum_{i=1}^{m_{\text{train}}} x^{(i)}_{j} \cdot dz^{(i)} \\
&db = \frac{\partial J(\mathbf{w}, b)}{\partial b} = \frac{1}{m_{\text{train}}}\sum_{i=1}^{m_{\text{train}}} \frac{\partial \mathcal{L}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial b} = \frac{1}{m_{\text{train}}}\sum_{i=1}^{m_{\text{train}}} dz^{(i)}\\
\end{align}
$$



### 2.7. Vectorizing Logistic Regression’s Gradient Output

$$
\begin{align}
&d\mathbf{Z} = \mathbf{A} - \mathbf{Y} \\
&d\mathbf{w} = \frac{1}{m_{\text{train}}} \mathbf{X}(d\mathbf{Z})^{\top}\\
&db = \frac{1}{m_{\text{train}}}\sum_{i=1}^{m_{\text{train}}} dz^{(i)}
\end{align}
$$



## 3. Shallow Neural Networks

### 3.1. L-Layer Neural Networks

* **Input layer** $$\mathbf{a}^{[0](i)}$$

* **Hidden layers** $$\mathbf{a}^{[l](i)}$$, $$1 \leq l < L$$

* **Output Layer** $$\mathbf{a}^{[L](i)}$$

Given single training example,
$$
\begin{align}
\mathbf{z}^{[l](i)} &= \mathbf{W}^{[l]} \mathbf{a}^{[l-1](i)} + \mathbf{b}^{[l]} \\
\mathbf{a}^{[l](i)} &= \operatorname{g}^{[l]}(\mathbf{z}^{[l](i)})
\end{align}
$$
where $$\mathbf{a}^{[l-1](i)} \in \mathbb{R}^{n_{l-1}}$$, $$\mathbf{a}^{[l](i)}, \mathbf{z}^{[l](i)}, \mathbf{b}^{[l]} \in \mathbb{R}^{n_{l}}$$, $$\mathbf{W}^{[l]} \in \mathbb{R}^{n_{l} \times n_{l-1}}$$.

Given $$m$$ training examples,
$$
\begin{align}
\mathbf{Z}^{[l]} &= \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]} \\
\mathbf{A}^{[l]} &= \operatorname{g}^{[l]}(\mathbf{Z}^{[l]})
\end{align}
$$
where $$\mathbf{A}^{[l]} = [\mathbf{a}^{[l](1)}, \mathbf{a}^{[l](2)}, \dots, \mathbf{a}^{[l](m)}]$$, $$\mathbf{Z}^{[l]} = [\mathbf{z}^{[l](1)}, \mathbf{z}^{[l](2)}, \dots, \mathbf{z}^{[l](m)}]$$, $$\mathbf{A}^{[l]}, \mathbf{Z}^{[l]} \in \mathbb{R}^{n_{l} \times m}$$.



### 3.2. Activation Functions

* **Sigmoid**

  When using binary classification, `Sigmoid` is a very natural choice for the output layer.
  $$
  a = \operatorname{Sigmoid}(z) = \sigma(z) = \frac{1}{1 + e^{-z}}
  $$


* **tanh**

  `tanh` kind of has the effect of centering the data so that the mean of the data is closer to 0. We could use it for the hidden layers.
  $$
  a = \operatorname{tanh}(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}
  $$


* **Rectified Linear Unit** (**ReLU**)

  One of the downsides of both `Sigmoid` and `tanh` is that if $$z$$ is either very large or very small, then the slope of the function becomes very small, which can slow down gradient descent.

  The derivative of `ReLU` is 1, so long as $$z$$ is positive. So our neural network will often learn much faster.

  Technically, the derivative when $$z$$ is exactly 0 is not well defined. In practical, we could pretend the derivative is either 1 or 0, when $$z$$ is equal to 0, and then it kind of works just fine.

  One disadvantage of `ReLU` is that the derivative is equal to zero, when $$z$$ is negative. In practice, this works just fine, enough of our hidden units will have $$z$$ greater than 0. So learning can still be quite fast for most training examples.
  $$
  a = \operatorname{ReLU}(z) = \max (0, z)
  $$



* **Leaky ReLU**
  $$
  a = \operatorname{LReLU}(z) = \max (0.01z, z)
  $$



> **The most commonly used activation function is `ReLU`.**



### 3.3. Why Non-Linear Activation Functions?

### 3.4. Derivatives of Activation Functions

### 3.5. Gradient Descent for Neural Networks

$$
\begin{align}
\mathbf{W}^{[l]} &= \mathbf{W}^{[l]} - \alpha \cdot d\mathbf{W}^{[l]} \\
\mathbf{b}^{[l]} &= \mathbf{b}^{[l]} - \alpha \cdot d\mathbf{b}^{[l]}
\end{align}
$$

Given single training example,
$$
\begin{align}
&d\mathbf{a}^{[l](i)} = (\mathbf{W}^{[l+1]})^{\top} d\mathbf{z}^{[l+1](i)} \\
&d\mathbf{z}^{[l](i)} = d\mathbf{a}^{[l](i)} \odot \operatorname{g}^{[l]\prime} (\mathbf{z}^{[l](i)}) \\
&d\mathbf{W}^{[l]} = d\mathbf{z}^{[l](i)}(\mathbf{a}^{[l-1](i)})^{\top}\\
&d\mathbf{b}^{[l]} = d\mathbf{z}^{[l](i)}
\end{align}
$$
where $$\odot$$ is element-wise multiplication.

Given $$m$$ training examples,
$$
\begin{align}
&d\mathbf{A}^{[l]} = (\mathbf{W}^{[l+1]})^{\top} d\mathbf{Z}^{[l+1]} \\
&d\mathbf{Z}^{[l]} = d\mathbf{A}^{[l]} \odot \operatorname{g}^{[l]\prime} (\mathbf{Z}^{[l]}) \\
&d\mathbf{W}^{[l]} = \frac{1}{m} d\mathbf{Z}^{[l]}(\mathbf{A}^{[l-1]})^{\top}\\
&d\mathbf{b}^{[l]} = \frac{1}{m}\sum_{i=1}^{m} d\mathbf{z}^{[l](i)}
\end{align}
$$



### 3.6. Backpropagation Intuition

### 3.7. Random Initialization

For a neural network, it's important to initialize the weights to **very small random** values. If we initialize the weights to all zeros and then apply gradient descent, all of our hidden units are symmetric (symmetry breaking problem). So that's not helpful, because we want the different hidden units to compute different functions.

It turns out initializing the biases to all zeros is actually okay.



## 4. Deep Neural Networks

### 4.1. Deep L-Layer Neural Networks

### 4.2. Why Deep Representations?

### 4.3. Building Blocks of Deep Neural Networks

### 4.4. Forward and Backward Propagation

### 4.5. Parameters vs. Hyperparameters

Applied deep learning is a very **empirical** process.
