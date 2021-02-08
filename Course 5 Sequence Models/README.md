# [Course 5 Sequence Models](https://www.coursera.org/learn/nlp-sequence-models?specialization=deep-learning)

## 1. Recurrent Neural Networks

### 1.1. Sequence Models

* **Sequence Generation**

* **Sentiment Classification**

* **Name Entity Recognition**

* **Machine Translation**

* **Speech Recognition**

* **Video Activity Recognition**

* **DNA Sequence Analysis**



### 1.2. Recurrent Neural Networks

**Name Entity Recognition**

Given a single example
$$
(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})
$$
where $$\mathbf{x}^{(i)} = [\mathbf{x}^{(i)<1>}, \mathbf{x}^{(i)<2>}, \dots, \mathbf{x}^{(i)<T_{x}^{(i)}>}]$$, if $$\mathbf{x}^{(i)<t>}$$ is one-hot encoded, then $$\mathbf{x}^{(i)<t>} \in \mathbb{R}^{|V|}$$, $$1 \leq t \leq T_{x}^{(i)}$$, $$\mathbf{x}^{(i)} \in \mathbb{R}^{|V| \times T_{x}^{(i)}}$$; $$\mathbf{y}^{(i)} = [y^{(i)<1>}, y^{(i)<2>}, \dots, y^{(i)<T_{y}^{(i)}>}]^{\top}$$, $$y^{(i)<t>} \in \{0, 1\}$$, $$1 \leq t \leq T_{y}^{(i)}$$, $$\mathbf{y}^{(i)} \in \mathbb{R}^{T_{y}^{(i)}}$$; and $$T_{x}^{(i)}=T_{y}^{(i)}$$.

We might introduce `<UNK>` to represent Out-of-Vocabulary (OOV) words.

#### 1.2.1. Forward Propagation

$$
\begin{align}
\mathbf{a}^{(i)<t>}  &= g_{a}(\mathbf{W}_{aa} \mathbf{a}^{(i)<t-1>} + \mathbf{W}_{ax} \mathbf{x}^{(i)<t>} + \mathbf{b}_{a}) \\
y^{(i)<t>} &= g_{y}(\mathbf{W}_{ya} \mathbf{a}^{(i)<t>}  + b_{y}) \\
\end{align}
$$

where we could initialize $$\mathbf{a}^{<0>}$$ to zeros, $$\mathbf{a}^{(i)<t>}, \mathbf{b}_{a} \in \mathbb{R}^{n_{h}}$$, $$\mathbf{W}_{aa} \in \mathbb{R}^{n_{h} \times n_{h}}$$, $$\mathbf{W}_{ax} \in \mathbb{R}^{n_{h} \times |V|}$$, $$\mathbf{W}_{ya} \in \mathbb{R}^{1 \times n_{h}}$$, $$g_{a}$$ can be `ReLU` or `tanh` and $$g_{y}$$ is `Sigmoid`.

Simplify the notation with concatenation
$$
\begin{align}
\mathbf{a}^{(i)<t>}  &= g_{a}(\mathbf{W}_{a} [\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] + \mathbf{b}_{a}) \\
y^{(i)<t>} &= g_{y}(\mathbf{W}_{y} \mathbf{a}^{(i)<t>}  + b_{y}) \\
\end{align}
$$
where $$\mathbf{W}_{a} = [\mathbf{W}_{aa}, \mathbf{W}_{ax}]$$, $$\mathbf{W}_{a} \in \mathbb{R}^{n_{h} \times (n_{h}+|V|)}$$, $$[\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] = \mathbf{a}^{(i)<t-1>} \oplus \mathbf{x}^{(i)<t>} = [\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}]^{\top}$$, $$[\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] \in \mathbb{R}^{n_{h}+|V|}$$, $$\mathbf{W}_{y} = \mathbf{W}_{ya} \in \mathbb{R}^{1 \times n_{h}}$$. 

#### 1.2.2. Backpropagation

Loss function
$$
\begin{align}
\mathcal{L}^{<t>}(\hat{y}^{(i)<t>}, y^{(i)<t>}) &= -(y^{(i)<t>} \log \hat{y}^{(i)<t>} + (1 - y^{(i)<t>}) \log (1 - \hat{y}^{(i)<t>})) \\
\mathcal{L}(\hat{\mathbf{y}}^{(i)}, \mathbf{y}^{(i)}) &= \sum_{t=1}^{T_{y}^{(i)}} \mathcal{L}^{<t>}(\hat{y}^{(i)<t>}, y^{(i)<t>})
\end{align}
$$


### 1.3. Different Types of RNNs

* One-to-many, e.g., Sequence Generation
* Many-to-one, e.g., Sentiment Classification
* Many-to-many ($$T_{x}^{(i)}=T_{y}^{(i)}$$), e.g., Name Entity Recognition
* Many-to-many ($$T_{x}^{(i)} \neq T_{y}^{(i)}$$), e.g., Machine Translation, Speech Recognition, etc.



### 1.4. Language Model and Sequence Generation

#### 1.4.1. Language Model

**Forward Propagation**

Given a single example $$\mathbf{x}^{(i)} = [\mathbf{x}^{(i)<1>}, \mathbf{x}^{(i)<2>}, \dots, \mathbf{x}^{(i)<T_{x}^{(i)}>}]$$ in a large enogh corpus, assuming $$\mathbf{y}^{(i)<t>} = \mathbf{x}^{(i)<t>}$$, $$1 \leq t \leq T_{x}^{(i)}$$, $$T_{y}^{(i)} = T_{x}^{(i)}$$,
$$
\begin{align}
\mathbf{a}^{(i)<t>}  &= g_{a}(\mathbf{W}_{a} [\mathbf{a}^{(i)<t-1>}, \mathbf{y}^{(i)<t-1>}] + \mathbf{b}_{a}) \\
\hat{\mathbf{y}}^{(i)<t>} &= g_{y}(\mathbf{W}_{y} \mathbf{a}^{(i)<t>}  + \mathbf{b}_{y}) \\
\end{align}
$$
where we could initialize $$\mathbf{a}^{<0>}$$, $$\mathbf{y}^{(i)<0>}$$ to zeros, $$\mathbf{a}^{(i)<t>}, \mathbf{b}_{a} \in \mathbb{R}^{n_{h}}$$, $$\mathbf{W}_{a} \in \mathbb{R}^{n_{h} \times (n_{h}+|V|)}$$,  $$\mathbf{y}^{(i)<t>}, \hat{\mathbf{y}}^{(i)<t>}, \mathbf{b}_{y} \in \mathbb{R}^{|V|}$$, $$[\mathbf{a}^{(i)<t-1>}, \mathbf{y}^{(i)<t-1>}] \in \mathbb{R}^{n_{h}+|V|}$$, $$\mathbf{W}_{y} \in \mathbb{R}^{|V| \times n_{h}}$$, $$g_{a}$$ can be `ReLU` or `tanh` and $$g_{y}$$ is `Softmax`. And $$\hat{\mathbf{y}}^{(i)<t>}$$ represents the distribution of the probability $$P(\mathbf{y}^{(i)<t>}|\mathbf{y}^{(i)<t-1>}, \dots, \mathbf{y}^{(i)<1>})$$.

We might introduce `<EOS>` to indicate the end of the sequence. So $$\hat{\mathbf{y}}^{(i)<T_{y}^{(i)}>}$$ represents the distribution of the probability $$P(\mathbf{y}_{\text{<EOS>}}^{(i)<T_{y}^{(i)}>}|\mathbf{y}^{(i)<T_{y}^{(i)}-1>}, \dots, \mathbf{y}^{(i)<1>})$$, where we actually have $$\mathbf{y}^{(i)<t>} = \mathbf{x}^{(i)<t>}$$, $$1 \leq t \leq T_{x}^{(i)}$$,  $$T_{y}^{(i)} = T_{x}^{(i)} + 1$$.

**Backpropagation**

Loss function
$$
\begin{align}
\mathcal{L}^{<t>}(\hat{\mathbf{y}}^{(i)<t>}, \mathbf{y}^{(i)<t>}) &= - \sum_{j=1}^{|V|} y_{j}^{(i)<t>} \log \hat{y}_{j}^{(i)<t>} \\
\mathcal{L}(\hat{\mathbf{y}}^{(i)}, \mathbf{y}^{(i)}) &= \sum_{t=1}^{T_{y}^{(i)}} \mathcal{L}^{<t>}(\hat{\mathbf{y}}^{(i)<t>}, \mathbf{y}^{(i)<t>})
\end{align}
$$
Minimizing the loss corresponds to maximizing the log of the probability (from Language Model)
$$
\begin{align}
P(\mathbf{y}^{(i)<t>}, \mathbf{y}^{(i)<t-1>}, \dots, \mathbf{y}^{(i)<1>}) = &P(\mathbf{y}^{(i)<t>}|\mathbf{y}^{(i)<t-1>}, \dots, \mathbf{y}^{(i)<1>}) \\
&\cdot P(\mathbf{y}^{(i)<t-1>}|\mathbf{y}^{(i)(t-2)}, \dots, \mathbf{y}^{(i)<1>}) \\
&\cdots \\
&\cdot P(\mathbf{y}^{(i)<2>}|\mathbf{y}^{(i)<1>}) \\
&\cdot P(\mathbf{y}^{(i)<1>}) 
\end{align}
$$

#### 1.4.2. Sequence Generation

Now, we could sample a novel sequence from a trained RNN, we initialize $$\mathbf{a}^{<0>}$$, $$\mathbf{y}^{<0>}$$ to zeros, then $$\mathbf{y}^{<t>}$$ is sampled from the probability distribution $$\hat{\mathbf{y}}^{<t>}$$, $$t \geq 1$$, but avoid `<UNK>` being sampled, iterate until `<EOS>` is sampled.

**Word-Level vs. Character-Level Language Model**



### 1.5. Vanishing and Exploding Gradients with RNNs

**Basic RNNs are not very good at capturing very long-term dependencies.** They usually run into vanishing and exploding gradients problem. 

#### 1.5.1. Vanishing Gradients with RNNs

Comparing to exploding gradients, vanishing gradients is much harder to solve.

#### 1.5.2. Exploding Gradients with RNNs

If the derivatives do explode or we see `NaN`, one solution to it is to apply gradient clipping, i.e., when a gradient vector is bigger than some threshold, re-scale it so that it is not too big. **It turns out gradient clipping is a relatively robust solution.**



### 1.6. Gated Recurrent Unit (GRU)

**Simplified GRU**
$$
\begin{align}
\mathbf{\Gamma}_{u}^{(i)<t>} &= \sigma(\mathbf{W}_{u} [\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] + \mathbf{b}_{u}) \\
\tilde{\mathbf{c}}^{(i)<t>} &= g(\mathbf{W}_{c} [\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] + \mathbf{b}_{c}) \\
\mathbf{c}^{(i)<t>} &= \mathbf{\Gamma}_{u}^{(i)<t>} \odot \tilde{\mathbf{c}}^{(i)<t>} + (1 - \mathbf{\Gamma}_{u}^{(i)<t>}) \odot \mathbf{c}^{(i)<t-1>}\\
\mathbf{a}^{(i)<t>} &= \mathbf{c}^{(i)<t>}
\end{align}
$$
where $$\odot$$ is element-wise multiplication, $$\mathbf{a}^{(i)<t>}, \mathbf{c}^{(i)<t>}, \mathbf{\Gamma}_{u}^{(i)<t>}, \mathbf{b}_{u}, \tilde{\mathbf{c}}^{(i)}, \mathbf{b}_{c} \in \mathbb{R}^{n_{h}}$$, $$\mathbf{W}_{u}, \mathbf{W}_{c} \in \mathbb{R}^{n_{h} \times (n_{h}+|V|)}$$, $$[\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] \in \mathbb{R}^{n_{h}+|V|}$$.

**GRU**
$$
\begin{align}
\mathbf{\Gamma}_{u}^{(i)<t>} &= \sigma(\mathbf{W}_{u} [\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] + \mathbf{b}_{u}) \\
\mathbf{\Gamma}_{r}^{(i)<t>} &= \sigma(\mathbf{W}_{r} [\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] + \mathbf{b}_{r}) \\
\tilde{\mathbf{c}}^{(i)<t>} &= g(\mathbf{W}_{c} [\mathbf{\Gamma}_{r}^{(i)<t>} \odot \mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] + \mathbf{b}_{c}) \\
\mathbf{c}^{(i)<t>} &= \mathbf{\Gamma}_{u}^{(i)<t>} \odot \tilde{\mathbf{c}}^{(i)<t>} + (1 - \mathbf{\Gamma}_{u}^{(i)<t>}) \odot \mathbf{c}^{(i)<t-1>}\\
\mathbf{a}^{(i)<t>} &= \mathbf{c}^{(i)<t>}
\end{align}
$$
where $$\odot$$ is element-wise multiplication, $$\mathbf{a}^{(i)<t>}, \mathbf{c}^{(i)<t>}, \mathbf{\Gamma}_{u}^{(i)<t>}, \mathbf{b}_{u}, \mathbf{\Gamma}_{r}^{(i)}, \mathbf{b}_{r}, \tilde{\mathbf{c}}^{(i)}, \mathbf{b}_{c} \in \mathbb{R}^{n_{h}}$$, $$\mathbf{W}_{u}, \mathbf{W}_{r}, \mathbf{W}_{c} \in \mathbb{R}^{n_{h} \times (n_{h}+|V|)}$$, $$[\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}], [\mathbf{\Gamma}_{r}^{(i)<t>} \odot \mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] \in \mathbb{R}^{n_{h}+|V|}$$.



### 1.7. Long Short Term Memory (LSTM) 

**Peephole Connection**
$$
\begin{align}
\mathbf{\Gamma}_{u}^{(i)<t>} &= \sigma(\mathbf{W}_{u} [\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] + \mathbf{b}_{u}) \\
\mathbf{\Gamma}_{f}^{(i)<t>} &= \sigma(\mathbf{W}_{f} [\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] + \mathbf{b}_{f}) \\
\mathbf{\Gamma}_{o}^{(i)<t>} &= \sigma(\mathbf{W}_{o} [\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] + \mathbf{b}_{o}) \\
\tilde{\mathbf{c}}^{(i)<t>} &= g(\mathbf{W}_{c} [\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] + \mathbf{b}_{c}) \\
\mathbf{c}^{(i)<t>} &= \mathbf{\Gamma}_{u}^{(i)<t>} \odot \tilde{\mathbf{c}}^{(i)<t>} + \mathbf{\Gamma}_{f}^{(i)<t>} \odot \mathbf{c}^{(i)<t-1>}\\
\mathbf{a}^{(i)<t>} &= \mathbf{\Gamma}_{o}^{(i)<t>} \odot \mathbf{c}^{(i)<t>}
\end{align}
$$
where $$\odot$$ is element-wise multiplication, $$\mathbf{a}^{(i)<t>}, \mathbf{c}^{(i)<t>}, \mathbf{\Gamma}_{u}^{(i)<t>}, \mathbf{b}_{u}, \mathbf{\Gamma}_{f}^{(i)}, \mathbf{b}_{f}, \mathbf{\Gamma}_{o}^{(i)}, \mathbf{b}_{o}, \tilde{\mathbf{c}}^{(i)}, \mathbf{b}_{c} \in \mathbb{R}^{n_{h}}$$, $$\mathbf{W}_{u}, \mathbf{W}_{f}, \mathbf{W}_{o}, \mathbf{W}_{c} \in \mathbb{R}^{n_{h} \times (n_{h}+|V|)}$$, $$[\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] \in \mathbb{R}^{n_{h}+|V|}$$.

Because $$\mathbf{a}^{(i)<t>} \neq \mathbf{c}^{(i)<t>}$$, $$\mathbf{c}^{(i)<t-1>}$$ may have effect on $$\mathbf{\Gamma}_{u}^{(i)<t>}$$, $$\mathbf{\Gamma}_{f}^{(i)<t>}$$, $$\mathbf{\Gamma}_{o}^{(i)<t>}$$. If take it into consideration,
$$
\begin{align}
\mathbf{\Gamma}_{u}^{(i)<t>} &= \sigma(\mathbf{W}_{u} [\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}, \mathbf{c}^{(i)<t-1>}] + \mathbf{b}_{u}) \\
\mathbf{\Gamma}_{f}^{(i)<t>} &= \sigma(\mathbf{W}_{f} [\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}, \mathbf{c}^{(i)<t-1>}] + \mathbf{b}_{f}) \\
\mathbf{\Gamma}_{o}^{(i)<t>} &= \sigma(\mathbf{W}_{o} [\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}, \mathbf{c}^{(i)<t-1>}] + \mathbf{b}_{o}) \\
\tilde{\mathbf{c}}^{(i)<t>} &= g(\mathbf{W}_{c} [\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] + \mathbf{b}_{c}) \\
\mathbf{c}^{(i)<t>} &= \mathbf{\Gamma}_{u}^{(i)<t>} \odot \tilde{\mathbf{c}}^{(i)<t>} + \mathbf{\Gamma}_{f}^{(i)<t>} \odot \mathbf{c}^{(i)<t-1>}\\
\mathbf{a}^{(i)<t>} &= \mathbf{\Gamma}_{o}^{(i)<t>} \odot \mathbf{c}^{(i)<t>}
\end{align}
$$
where $$\odot$$ is element-wise multiplication, $$\mathbf{a}^{(i)<t>}, \mathbf{c}^{(i)<t>}, \mathbf{\Gamma}_{u}^{(i)<t>}, \mathbf{b}_{u}, \mathbf{\Gamma}_{f}^{(i)}, \mathbf{b}_{f}, \mathbf{\Gamma}_{o}^{(i)}, \mathbf{b}_{o}, \tilde{\mathbf{c}}^{(i)}, \mathbf{b}_{c} \in \mathbb{R}^{n_{h}}$$, $$\mathbf{W}_{u}, \mathbf{W}_{f}, \mathbf{W}_{o}, \mathbf{W}_{c} \in \mathbb{R}^{n_{h} \times (n_{h}+|V|+n_{h})}$$, $$[\mathbf{a}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}, \mathbf{c}^{(i)<t-1>}] \in \mathbb{R}^{n_{h}+|V|+n_{h}}$$.



**GRU vs. LSTM**

The advantage of GRU is that it's a simpler model, so it is actually easier to build a much bigger network, it only has two gates, computationally, it runs a bit faster. LSTM is more powerful and more effective since it has three gates instead of two.

If we have to pick one, most people today will still use LSTM as the default first thing to try.



### 1.8. Bidirectional RNNs

$$
\begin{align}
\overrightarrow{\mathbf{a}}^{(i)<t>}  &= g_{\overrightarrow{a}}(\mathbf{W}_{\overrightarrow{a}} [\overrightarrow{\mathbf{a}}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}] + \mathbf{b}_{\overrightarrow{a}}) \\
\overleftarrow{\mathbf{a}}^{(i)<t>}  &= g_{\overleftarrow{a}}(\mathbf{W}_{\overleftarrow{a}} [\overleftarrow{\mathbf{a}}^{(i)<t+1>}, \mathbf{x}^{(i)<t>}] + \mathbf{b}_{\overleftarrow{a}}) \\
y^{(i)<t>} &= g_{y}(\mathbf{W}_{y} [\overrightarrow{\mathbf{a}}^{(i)<t>}, \overleftarrow{\mathbf{a}}^{(i)<t>}]  + b_{y}) \\
\end{align}
$$

where $$\overrightarrow{\mathbf{a}}^{(i)<t>}, \mathbf{b}_{\overrightarrow{a}}, \overleftarrow{\mathbf{a}}^{(i)<t>}, \mathbf{b}_{\overleftarrow{a}} \in \mathbb{R}^{n_{h}}$$, $$\mathbf{W}_{\overrightarrow{a}}, \mathbf{W}_{\overleftarrow{a}} \in \mathbb{R}^{n_{h} \times (n_{h}+|V|)}$$, $$[\overrightarrow{\mathbf{a}}^{(i)<t-1>}, \mathbf{x}^{(i)<t>}], [\overleftarrow{\mathbf{a}}^{(i)<t+1>}, \mathbf{x}^{(i)<t>}] \in \mathbb{R}^{n_{h}+|V|}$$, $$\mathbf{W}_{y} \in \mathbb{R}^{1 \times (n_{h}+n_{h})}$$, $$[\overrightarrow{\mathbf{a}}^{(i)<t-1>}, \overleftarrow{\mathbf{a}}^{(i)<t+1>}] \in \mathbb{R}^{n_{h}+n_{h}}$$. 

**The disadvantage of bidirectional RNNs is that we do need the entire sequence of data before we are able to make predictions anywhere.** 



### 1.9. Deep RNNs

$$
\mathbf{a}^{(i)[l]<t>}  = g_{a}^{[l]}(\mathbf{W}_{a}^{[l]} [\mathbf{a}^{(i)[l]<t-1>}, \mathbf{a}^{(i)[l-1]<t>}] + \mathbf{b}_{a}^{[l]})
$$



## 2. Natural Language Processing & Word Embeddings

### 2.1. Introduction to Word Embeddings

#### 2.1.1. Word Representation

**One-Hot Encoding**

Let's take any pair vectors, the dot product of them is zero. We can't know the similarity of any two words.

**Featurized Representation: Word Embedding**

**Transfer Learning and Word Embedding**

* Learn word embeddings from large enough corpus (~100B words) or use pre-trained word embeddings.
* Transfer embedding to a new task with much smaller Train set (e.g., ~100K words).
* Optional: fine-tune word embeddings using new data, which may help only when Train set is large enough.

**Cosine Similarity**
$$
\operatorname{cos}(\theta) = \frac{\mathbf{u}^{\top}\mathbf{v}}{\|\mathbf{u}\|_{2} \cdot \|\mathbf{v}\|_{2}}
$$

#### 2.1.2. Embedding Matrix

$$
\mathbf{e}_{w} = \mathbf{E}\mathbf{o}_{w}
$$

where $$\mathbf{e}_{w} \in \mathbb{R}^{n_{e}}$$, $$\mathbf{E} \in \mathbb{R}^{n_{e} \times |V|}$$, $$\mathbf{o}_{w} \in \mathbb{R}^{|V|}$$. The embedding vector $$\mathbf{e}_{w}$$ is the $$w$$-th column of the embedding matrix $$\mathbf{E}$$.



### 2.2. Learning Word Embeddings

**Choose context and Target Pairs**



### 2.3. Word2Vec

#### 2.3.1. Skip-Gram

Randomly pick a word to be the context word, then randomly pick another word to be target word within some window.

Given context $$c$$ and target $$t$$, 
$$
\mathbf{e}_{c} = \mathbf{E}_{c}\mathbf{o}_{c} \\
\hat{\mathbf{y}} = \operatorname{Softmax}(\mathbf{\Theta}_{t}^{\top}\mathbf{e}_{c})
$$
where $$\mathbf{o}_{c}, \hat{\mathbf{y}} \in \mathbb{R}^{|V|}$$, $$\mathbf{E}_{c} = \mathbf{\Theta}_{t} \in \mathbb{R}^{n_{e} \times |V|}$$, $$\mathbf{e}_{c} \in \mathbb{R}^{n_{e}}$$. And $$\hat{\mathbf{y}}$$ represents the distribution of the probability $$P(t|c)$$.
$$
\hat{y}_{j} = \frac{e^{\mathbf{\theta}_{j}^{\top}\mathbf{e}_{c}}}{\sum_{j=1}^{|V|}e^{\mathbf{\theta}_{j}^{\top}\mathbf{e}_{c}}}
$$
where $$\mathbf{\theta}_{t}$$ is the parameter corresponds to target $$t$$, the $$t$$-th column of the hidden layer weight matrix $$\mathbf{\Theta}_{t}$$. Also, 
$$
\sum_{j=1}^{|V|} \hat{y}_{j} = 1
$$
Loss function
$$
\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = - \sum_{j=1}^{|V|} y_{j} \log \hat{y}_{j}
$$
The key problem with Skip-Gram is that the `Softmax` step is very computationally expensive.

> How to sample context $$c$$?
>
> We don't want Train set to be dominated by those most common words.

#### 2.3.2. CBOW

CBOW uses the surrounding words to try to predict the middle word while Skip-Gram uses the middle word to try to predict the surrounding words.

#### 2.3.3. Negative Sampling

Sample a context word, then randomly pick another word to be target word within some window (one positive example). Take the same context word, pick another random word to be target word from the dictionary, that probably won't be associated with the context word (one negative example). Sample one positive example and $$k$$ negative examples with the same context word. 

> How to choose $$k$$?
>
> $$k$$ is 5 to 20 for smaller data sets. If we have a very large data set, then chose $$k$$ to be smaller, e.g., $$k$$ equals 2 to 5 for larger data sets.

$$
\hat{y} = P(y = 1 | c, t) = \sigma(\mathbf{\theta}_{t}^{\top}\mathbf{e}_{c})
$$

> How to sample negative examples?
> $$
> P(w_{j}) = \frac{\operatorname{count}(w_{j})}{\sum_{j=1}^{|V|} \operatorname{count}(w_{j})}
> $$

### 2.4. GloVe

**Global Vectors (GloVe) for Word Representation**

### 2.5. Application using Word Embeddings: Sentiment Classification

### 2.6. Debiasing Word Embeddings



## 3. Sequence-to-Sequence Models & Attention Mechanism

### 3.1. Encoder-Decoder Architectures

**Machine Translation (Conditional Language Model)**
$$
\max P(\mathbf{y}^{(i)<1>}, \mathbf{y}^{(i)<2>}, \dots, \mathbf{y}^{(i)<T_{y}^{(i)}>}|\mathbf{x}^{(i)<1>}, \mathbf{x}^{(i)<2>}, \dots, \mathbf{x}^{(i)<T_{x}^{(i)}>})
$$
where probably $$T_{y}^{(i)} \neq T_{x}^{(i)}$$.



### 3.2. Beam Search

**Why Not Greedy Search?**

#### 3.2.1. Beam Search

At step $$t$$ with the decoder, maintain top-$$B$$ generated sequence $$\mathbf{y}^{<1>}, \mathbf{y}^{<2>}, \dots, \mathbf{y}^{<t>}$$ ranked by the conditional probability
$$
P(\mathbf{y}^{<1>}, \mathbf{y}^{<2>}, \dots, \mathbf{y}^{<t>}|\mathbf{x}) = \prod_{i=1}^{t}P(\mathbf{y}^{<i>}|\mathbf{x}, \mathbf{y}^{<i-1>}, \dots, \mathbf{y}^{<2>}, \mathbf{y}^{<1>})
$$
where $$B$$ is the beam width, $$\mathbf{y}^{<t>}$$ is sampled from the probability distribution $$\hat{\mathbf{y}}^{<t>}$$, $$t \geq 1$$. When $$B=1$$, beam search is identical to greedy search.

#### 3.2.2. Refinements to Beam Search

**Length Normalization Logarithm Likelihood Object**
$$
\arg \max \log P(\mathbf{y}^{<1>}, \mathbf{y}^{<2>}, \dots, \mathbf{y}^{<t>}|\mathbf{x}) = \arg \max \frac{1}{t}\sum_{i=1}^{t}\log P(\mathbf{y}^{<i>}|\mathbf{x}, \mathbf{y}^{<i-1>}, \dots, \mathbf{y}^{<2>}, \mathbf{y}^{<1>})
$$
In practice, we usually implement a softer approach
$$
\arg \max \frac{1}{t^{\alpha}}\sum_{i=1}^{t}\log P(\mathbf{y}^{<i>}|\mathbf{x}, \mathbf{y}^{<i-1>}, \dots, \mathbf{y}^{<2>}, \mathbf{y}^{<1>})
$$
where $$\alpha \in [0, 1]$$.

#### 3.2.3. Error Analysis in Beam Search

### 3.3. Bleu Score

### 3.4. Attention Model

**Decoder**
$$
\begin{align}
\mathbf{s}^{<t>}  &= g_{s}(\mathbf{W}_{s} [\mathbf{s}^{<t-1>}, \mathbf{y}^{<t-1>}, \mathbf{c}^{<t>}] + \mathbf{b}_{s}) \\
\hat{\mathbf{y}}^{<t>} &= g_{y}(\mathbf{W}_{y} \mathbf{s}^{<t>}  + \mathbf{b}_{y}) \\
\end{align}
$$
**Bidirectional Encoder**
$$
\mathbf{a}^{<t^{\prime}>} = [\overrightarrow{\mathbf{a}}^{<t^{\prime}>}, \overleftarrow{\mathbf{a}}^{<t^{\prime}>}]
$$
Context vector $$\mathbf{c}^{<t>}$$ is a weighted sum of those activations
$$
\mathbf{c}^{<t>} = \sum_{t^{\prime}=1}^{T_{x}} \alpha^{<t, t^{\prime}>} \mathbf{a}^{<t^{\prime}>}
$$
where $$\alpha^{<t, t^{\prime}>}$$ is the amount of attention $$\mathbf{y}^{<t>}$$ should pay to $$\mathbf{a}^{<t^{\prime}>}$$.
$$
\alpha^{<t, t^{\prime}>} = \frac{\exp (e^{<t, t^{\prime}>})}{\sum_{t^{\prime}=1}^{T_{x}} \exp (e^{<t, t^{\prime}>})}
$$
and
$$
\sum_{t^{\prime}=1}^{T_{x}} \alpha^{<t, t^{\prime}>} = 1
$$
where
$$
e^{<t, t^{\prime}>} = \operatorname{FNN}([\mathbf{s}^{<t-1>}, \mathbf{a}^{<t^{\prime}>}])
$$
is an alignment model which scores how well the inputs around step $$t^{\prime}$$ and the output at step $$t$$ match.



### 3.5. Audio Data: Speech Recognition

#### 3.5.1. Speech Recognition

#### 3.5.2. Trigger Word Detection

