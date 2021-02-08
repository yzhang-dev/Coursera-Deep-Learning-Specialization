# [Course 4 Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning)

## 1. Foundations of Convolutional Neural Networks

### 1.1. Computer Vision

* **Edge Detection**

* **Image Classification**

* **Object Detection**

* **Neural Style Transfer**

**Fully Connected Networks vs. Convolutional Neural Networks**

With that many parameters in fully connected networks, it's difficult to get enough data to prevent a neural network from overfitting. Besides, the computational and memory requirements are infeasible.



### 1.2. Edge Detection

**Vertical and Horizontal Edge Detection**

**Convolutions**
$$
\mathbf{x}_{\text{output}}  = \mathbf{x}_{\text{input}} * \mathbf{f}
$$
where $$\mathbf{x}_{\text{input}} \in \mathbb{R}^{n_{H} \times n_{W}}$$, $$\mathbf{f} \in \mathbb{R}^{f \times f}$$, $$\mathbf{x}_{\text{output}} \in \mathbb{R}^{(n_{H}-f+1) \times (n_{W}-f+1)}$$.

**Convolutional Neural Networks Intuition**

Learn the weights in the convolutional filters.



### 1.3. Padding

**Why Padding?**

* Each time we apply a convolutional operator, the image shrinks, i.e., $$n_{H^{\prime}}$$, $$n_{W^{\prime}}$$ decreases.

* A lot of the information from the edges is kind of thrown away.

**Convolutions with Padding**
$$
\mathbf{x}_{\text{output}}  = \operatorname{Pad}(\mathbf{x}_{\text{input}}) * \mathbf{f}
$$
where $$\mathbf{x}_{\text{input}} \in \mathbb{R}^{n_{H} \times n_{W}}$$, $$\mathbf{f} \in \mathbb{R}^{f \times f}$$, $$\mathbf{x}_{\text{output}} \in \mathbb{R}^{(n_{H}+2p-f+1) \times (n_{W}+2p-f+1)}$$.

**Valid and Same Convolutions**

* Valid: $$p=0$$, $$\mathbf{x}_{\text{output}} \in \mathbb{R}^{(n_{H}-f+1) \times (n_{W}-f+1)}$$.

* Same (pad so that output size is the same as input size): $$p = (f - 1) / 2$$, $$\mathbf{x}_{\text{output}} \in \mathbb{R}^{n_{H} \times n_{W}}$$.

By convention in computer vision, $$f$$ is usually odd. If $$f$$ is odd, then the same convolution gives a natural symmetric padding region, and the filter has a central position.



### 1.4. Strided Convolutions

Considering stride,
$$
\mathbf{x}_{\text{output}}  = \operatorname{Pad}(\mathbf{x}_{\text{input}}) * \mathbf{f}
$$
where $$\mathbf{x}_{\text{input}} \in \mathbb{R}^{n_{H} \times n_{W}}$$, $$\mathbf{f} \in \mathbb{R}^{f \times f}$$, $$\mathbf{x}_{\text{output}} \in \mathbb{R}^{\lfloor\frac{n_{H}+2p-f}{s}+1\rfloor \times \lfloor\frac{n_{W}+2p-f}{s}+1\rfloor}$$.



### 1.5. Convolutions over Volumes

$$
\mathbf{x}_{\text{output}}  = \operatorname{Stack}_{j=1}^{n_{C^{\prime}}} \operatorname{Pad}(\mathbf{x}_{\text{input}}) * \mathbf{f}_{j}
$$

where $$\mathbf{x}_{\text{input}} \in \mathbb{R}^{n_{H} \times n_{W} \times n_{C}}$$, $$\mathbf{f}_{j} \in \mathbb{R}^{f \times f \times n_{C}}$$, $$1 \leq j \leq n_{C^{\prime}}$$, $$n_{C^{\prime}}$$ represents the number of convolutional filters, $$\mathbf{x}_{\text{output}} \in \mathbb{R}^{\lfloor\frac{n_{H}+2p-f}{s}+1\rfloor \times \lfloor\frac{n_{W}+2p-f}{s}+1\rfloor \times n_{C^{\prime}}}$$.



### 1.6. 2D Convolution Layer

Given a mini-batch of size $m$,
$$
\operatorname{Conv2d}(\mathbf{A}^{[l]}) = \operatorname{Stack}_{j=1}^{n_{C}^{[l+1]}} (\operatorname{Pad}(\mathbf{A}^{[l]}) * \mathbf{F}_{j}^{[l]})
$$
where $$\mathbf{A}^{[l]} \in \mathbb{R}^{m \times n_{H}^{[l]} \times n_{W}^{[l]} \times n_{C}^{[l]}}$$, $$\mathbf{F}_{j}^{[l]} \in \mathbb{R}^{1 \times f \times f \times n_{C}^{[l]}}$$, $$1 \leq j \leq n_{C}^{[l+1]}$$, $$n_{C}^{[l+1]}$$ represents the number of convolutional filters, $$\operatorname{Conv2d}(\mathbf{A}^{[l]}) \in \mathbb{R}^{m \times \lfloor\frac{n_{H}^{[l]}+2p-f}{s}+1\rfloor \times \lfloor\frac{n_{W}^{[l]}+2p-f}{s}+1\rfloor \times n_{C}^{[l+1]}}$$.
$$
\begin{align}
\mathbf{Z}^{[l]}  &= \operatorname{Conv2d}(\mathbf{A}^{[l-1]}) + \mathbf{b}^{[l]} \\
\mathbf{A}^{[l]} &= g^{[l]}(\mathbf{Z}^{[l]})
\end{align}
$$
where $$\mathbf{A}^{[l-1]} \in \mathbb{R}^{m \times n_{H}^{[l-1]} \times n_{W}^{[l-1]} \times n_{C}^{[l-1]}}$$, $$\mathbf{A}^{[l]}, \mathbf{Z}^{[l]} \in \mathbb{R}^{m \times n_{H}^{[l]} \times n_{W}^{[l]} \times n_{C}^{[l]}}$$, $$\mathbf{b}^{[l]} \in \mathbb{R}^{1 \times 1 \times 1 \times n_{C}^{[l]}}$$



### 1.7. Pooling Layer

**Why Pooling Layer?**

ConvNets often **use pooling layers to reduce the height and width** of the representation, to speed the computation, as well as make the detected features more robust. 

**Max Pooling** 

**Average Pooling**

Pooling applies to each of channels independently. So **the number of channels remains unchanged**.

One thing to note about pooling is that there are no parameters to learn.

Max pooling is used much more in ConvNets than average pooling. Usually, we choose $$f=s=2$$, $$p=0$$.



### 1.8. CNN Example: LeNet-5

* **Convolution Layer（CONV）**

* **Pooling Layer（POOL）**

* **Fully Connected Layer（FC）**



**Image Classification: LeNet-5**

|                         | Activation Shape | Activation Size | Number of Parameters |
| :---------------------: | ---------------- | --------------- | -------------------- |
|        **Input**        | (32, 32, 3)      | 3072            | 0                    |
| **CONV1** (f = 5, s= 1) | (28, 28, 8)      | 6272            | 608                  |
|        **POOL1**        | (14, 14, 8)      | 1568            | 0                    |
| **CONV2** (f = 5, s= 1) | (10, 10, 16)     | 1600            | 3216                 |
|        **POOL2**        | (5, 5, 16)       | 400             | 0                    |
|         **FC3**         | (120, 1)         | 120             | 48120                |
|         **FC4**         | (84, 1)          | 84              | 10164                |
|       **Softmax**       | (10, 1)          | 10              | 850                  |

As we go deeper, usually, the height and width will decrease, whereas the number of channels will increase. The activation size tends to go down gradually as we go deeper. If it drops too quickly, that's usually not great for performance. 



### 1.9. Why Convolutions?

* **Parameter sharing**: A feature detector (convolutional filter) that’s useful in one part of the image is probably useful in another part of the image.
* **Sparsity of connection**: In each layer, each output value depends only on a small number of inputs.
* **Translation invariance**



### 1.10. 1D and 3D Generalizations



## 2. Deep Convolutional Models: Case Studies

### 2.1. Classic Networks

#### 2.1.1. LeNet-5

#### 2.1.2. AlexNet

#### 2.1.3. VGG-16



### 2.2. Residual Networks

#### 2.2.1. ResNets

**Residual Block with Skip Connection (Shortcut)**
$$
\begin{align}
\mathbf{Z}^{[l]}  &= \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]} \\
\mathbf{A}^{[l]} &= g^{[l]}(\mathbf{Z}^{[l]}) \\
\mathbf{Z}^{[l+1]}  &= \mathbf{W}^{[l+1]} \mathbf{A}^{[l]} + \mathbf{b}^{[l+1]} \\
\mathbf{A}^{[l+1]} &= g^{[l]}(\mathbf{Z}^{[l+1]} \oplus \mathbf{A}^{[l-1]})
\end{align}
$$
**Residual Networks vs. Plain Networks**

* In reality, the training error gets worse if we pick a plain network that's too deep.

* ResNets really help with the vanishing and exploding gradient problems and allow us to train much deeper neural networks without really hurting performance.

#### 2.2.2. Why Do ResNets Work?



### 2.3. Inception Network

#### 2.3.1. Networks in Networks and 1x1 Convolutions

One by one convolution operation is actually doing a pretty non-trivial operation and it **allows us to shrink the number of channels in volumes** (or keep it the same, or even increase it).

#### 2.3.2. Inception Network Motivation

#### 2.3.3. Inception Network



### 2.4. Practical Advices for Using ConvNets

#### 2.4.1. Using Open-Source Implementation

#### 2.4.2. Transfer Learning

#### 2.4.3. Data Augmentation

#### 2.4.4. State of Computer Vision



## 3. Object Detection

### 3.1. Object Localization

**Localization vs. Detection**

**Defining Target Label** 

* Verification (Binary Classification): Logistic Regression

* Localization: Boundary Box

* Multi-class Classification: Softmax Regression

### 3.2. Landmark Detection

### 3.3. Object Detection

**Sliding Windows Detection**



### 3.4. Convolutional Implementation of Sliding Windows

### 3.5. Bounding Box Predictions

### 3.6. Intersection Over Union

### 3.7. Non-Max Suppression

### 3.8. Anchor Boxes

### 3.9. YOLO Algorithm

### 3.10. Region Proposals



## 4. Face Recognition

### 4.1. Face Recognition

### 4.2. One Shot Learning

### 4.3. Siamese Network

### 4.4. Triplet Loss

### 4.5. Face Verification



## 5. Neural Style Transfer

### 5.1. Neural Style Transfer

### 5.2. What Are Deep ConvNets Learning?

### 5.3. Content and Style Cost Function

#### 5.3.1. Content Cost Function

#### 5.3.2. Style Cost Function

