---
layout: default
title: Transformer Building Blocks
---

# Chapter 1: Transformer Building Blocks

## 1.1 Queries, Keys, and Values in Attention Mechanisms

### Motivation

Let’s assume we have some data points:

$$
\mathcal{D} = \big\{ (x_1, y_1), (x_2, y_2), \dots, (x_n, y_n) \in \mathbb{R}^d, \quad d \geq 1\big\}.
$$

Given a new input $x$, how do we estimate $ y $?  
A classic nonparametric regression approach (Nadaraya–Watson) suggests, we can estimate $ y $ using a weighted sum of observed values:

$$
\hat{y} = \sum_{i=1}^{n} \alpha(x, x_i) \; y_i
$$

where $\alpha$ is  a [**kernel function**](https://github.com/KamilaKare/KamilaKare.github.io/blob/main/Formations/LLM%20for%20DS/Module%200/index.md) that determines how similar $x$ is to each $x_i$. The resulting $\hat{y}$ is a weighted average of the observed $y_i$ 
with weights depending on the similarity $\alpha(x, x_i)$. 


### Connecting to Attention

The **attention mechanism** extends this idea. Instead of input-output pairs $ (x_i, y_i) $, we define:

- **Keys $(\mathbf{k}_i)$**: Analogous to the $x_i$ in nonparametric regression, each token has a key vector that represents “where” it might be relevant, i.e. the locations where relevants tokens can be found.
- **Values $\mathbf{v}_i$**: Analogous to $y_i$, each token has a value vector that contains the actual content or information to be retrieved ( content associated with each key).
- **Query $\mathbf{q}$**: Analogous to the new input $x$. It's what we use to figure out which $\mathbf{k}_i$ are most relevant.

Some intuition might help here: for instance, in a regression setting, the query might correspond to the location where the regression should be carried out. The keys are the locations where past data was observed and the values are the (regression) values themselves.


Now, given a **database** 

$$\mathcal{D} = \big\{(\mathbf{k}_1, \mathbf{v}_1), (\mathbf{k}_2, \mathbf{v}_2), \dots, (\mathbf{k}_m, \mathbf{v}_m) \big\}$$

of key-value pairs. For a given query $\mathbf{q}$,
attention pools the values via weights $ \alpha(\mathbf{q}, \mathbf{k}_i)$

$$
\text{Attention}(\mathbf{q}, \mathcal{D}) = \sum_{i=1}^{m} \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i \tag{1}
$$

where $\alpha(\mathbf{q}, \mathbf{k}_i)$ are attention weights.

(1) is called **Attention Pooling**. The attention weights $\alpha(\mathbf{q}, \mathbf{k}_i)$ have specific properties:

- They are **nonnegative**:  $\alpha(\mathbf{q}, \mathbf{k}_i) \geq 0 $, The model never assigns negative importance.
- They form a **convex combination**: $ \sum_{i} \alpha(\mathbf{q}, \mathbf{k}_i) = 1 $.
- If all weights are **equal** $(\alpha = \frac{1}{m}$), we’re simply averaging all $\mathbf{v}_i$ equally, which is a naive baseline..

The beauty of attention lies in its ability to help the model learn to focus on the most relevant keys (and thus their values) based on the query, effectively performing a data-driven weighting akin to nonparametric smoothing.


This figure illustrates how the attention is computed in modern Transformer networks.

{% include image.html src="https://github.com/user-attachments/assets/709ee22a-9612-4bbc-a974-b2d3ccd73dbd" alt="Attention computation" caption="Attention computation." %}


This perspective helps us understand why attention mechanisms are so powerful: they **dynamically select the most relevant “neighbors”** in the dataset and **combine their values** based on similarity to the query.


### Attention Mechanism Formula

Given a set of **queries** $ \mathbf{Q} $, **keys** $ \mathbf{K} $, and **values** $ \mathbf{V} $, the attention mechanism computes an output as a weighted sum of values:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax} \left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V} \tag{2}
$$

where:

- $ \mathbf{Q} \in \mathbb{R}^{n_q \times d_k} $ is the **query matrix** (representing the search input).
- $ \mathbf{K} \in \mathbb{R}^{n_k \times d_k} $ is the **key matrix** (representing stored information).
- $ \mathbf{V} \in \mathbb{R}^{n_k \times d_v} $ is the **value matrix** (containing data to be retrieved).
- $ d_k $ is the **dimensionality of keys and queries**.
- $ \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} $ computes the **scaled dot-product similarity** between queries and keys.
- The **softmax function** ensures that the attention scores sum to 1, assigning different levels of importance to different values.

### Expanded Form

Considering Equations (1) and (2), as a result from identification, we have

$$
\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp \left( \frac{\mathbf{q} \cdot \mathbf{k}_i}{\sqrt{d_k}} \right)}{\sum_{j=1}^{m} \exp \left( \frac{\mathbf{q} \cdot \mathbf{k}_j}{\sqrt{d_k}} \right)}
$$

This defines a probability distribution over the values, focusing more on the most **relevant** ones.

## 1.2 Multi-Head Attention

In real-world applications, using a single attention mechanism may **limit the model’s ability** to capture diverse relationships within a sequence. A single attention head might focus on **short-range dependencies**, while another might capture **long-range dependencies**. To **enhance the model’s flexibility**, we need a mechanism that can learn **multiple aspects of attention** simultaneously.

To achieve this, instead of applying a **single attention pooling**, we transform the **queries, keys, and values** using **independently learned linear projections**. These projected representations are then **processed by multiple attention mechanisms (heads) in parallel**. Each head specializes in **extracting different types of information** from the input data.

Once all attention heads generate their outputs, these are **concatenated** and passed through another **learned linear transformation**. This final step integrates the information from all heads, producing a richer and more informative representation.

This approach is known as **Multi-Head Attention**, where each individual attention mechanism operates as an independent **head** within the overall framework. By leveraging multiple attention heads, the model gains the ability to focus on different aspects of the input, leading to **improved contextual understanding** and **enhanced learning efficiency**.

#### Multi-Head Attention Mechanism

1. Each query, key, and value undergoes a learned linear transformation:
   
   $$
   \mathbf{Q}_i = \mathbf{Q} \mathbf{W}_i^Q, \quad 
   \mathbf{K}_i = \mathbf{K} \mathbf{W}_i^K, \quad
   \mathbf{V}_i = \mathbf{V} \mathbf{W}_i^V
   $$

   where $ \mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V $ are the weight matrices for the **i-th attention head**.

2. Each **head** performs **scaled dot-product attention**:

   $$
   \text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i) = \text{softmax} \left( \frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_k}} \right) \mathbf{V}_i
   $$

3. The **outputs of all heads** are concatenated:

   $$
   \text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O
   $$

   where $ \mathbf{W}^O $ is a learned **output projection matrix**.

#### Benefits of Multi-Head Attention

- **Captures different types of dependencies** within a sequence.
- **Attends to information at multiple representation subspaces**.
- **Improves model robustness** by allowing different perspectives on data.

Multi-Head Attention is a **core component of the Transformer architecture**, enabling it to process contextual information efficiently. 

## 1.2 Self-Attention  
### Understanding Self-Attention
In deep learning, we often use **CNNs** or **RNNs** to encode sequences. However, with **attention mechanisms**, we can process sequences differently.  

Imagine feeding a sequence of tokens into an attention mechanism where each token has its **own query, key, and value**. When computing a token’s representation at the next layer:
- The token **attends** to all other tokens based on the compatibility of its **query** with their **keys**.
- Using these query-key scores, a **weighted sum** over the values is computed.
- This weighted sum forms the new representation of the token.  

In **self-attention**, the queries, keys, and values all come from the same sequence of tokens. In contrast, for tasks like machine translation, **cross-attention** occurs between two different sequences (e.g., the source sequence from the encoder and the target sequence from the decoder).

### Formula

Given a sequence of input tokens **$ x_1, x_2, \dots, x_n $**, where each **$ x_i \in \mathbb{R}^d $** for **$ 1 \leq i \leq n $**, self-attention computes an output sequence of the same length **$ y_1, y_2, \dots, y_n $**, where:

$$
\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \dots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d
$$

according to the definition of **attention pooling** (1).  

  

This mechanism is the foundation of **Transformers**, enabling powerful architectures like **GPT, BERT, and T5**.
By design, the attention mechanism provides a differentiable means of control by which a neural network can select elements from a set and to construct an associated weighted sum over representations.

---
## 1.2 Positional Encoding 
### Why Do We Need Positional Encoding?  

Unlike RNNs, which process input sequentially and inherently capture order information, the Transformer architecture processes all tokens in parallel. While this enables faster training and better scalability, it also means that Transformers have no built-in way to understand the order of tokens in a sequence.  

To address this, **Positional Encoding** is introduced to inject information about the relative or absolute positions of tokens within a sequence. This helps the model capture the sequence structure while maintaining the benefits of parallel computation.  

## 1.2.1 Absolute Positional Encoding

#### 1.2.1.1 Sinusoidal Encodings

Introduced by *Vaswani et al. (2017)*, **sinusoidal positional encodings** generate a deterministic set of vectors using sine and cosine functions of varying frequencies:

$$
PE_{\text{pos}, 2i} = \sin\Bigl(\frac{\text{pos}}{10000^{2i/d}}\Bigr), \quad
PE_{\text{pos}, 2i+1} = \cos\Bigl(\frac{\text{pos}}{10000^{2i/d}}\Bigr)
$$

- $\text{pos}$ = position index in the sequence.
- $d$ = dimensionality of the embedding.
- $i$ = dimension index (split into even and odd parts for sine and cosine).

**Why Sine and Cosine?**  
These formulas leverage sine and cosine functions to generate wave-like patterns that change with sequence positions. By applying sine to even indices and cosine to odd indices, they create a diverse set of features that effectively encode positional information across varying sequence lengths.

#### 1.2.1.2 Learned Absolute Position Embeddings

Instead of a fixed sinusoidal pattern, some models use a **trainable embedding vector** $\mathbf{p}_{\text{pos}}$ for each position $\text{pos}$. These vectors are typically initialized randomly and learned jointly with other model parameters.

---
## 1.2.3 Relative Positional Encoding

### 1.2.3.1 Concept

Rather than focusing on each token’s absolute position, **relative positional encoding** captures how far apart two tokens are. This can be crucial when local context is more relevant than exact indices. For example, in some tasks, token 10 and token 11 might have a stronger relationship than token 1 and token 10, purely due to proximity.

### 1.2.3.2 Mechanism

Often implemented by **modifying attention** with a term that depends on the relative distance between tokens. One simple approach:

$$
\text{Attention}(Q, K, V) \;=\;
\text{softmax}\Bigl(\frac{QK^\top + R}{\sqrt{d_k}}\Bigr)V,
$$

where $R$ is a learned matrix encoding pairwise distances between tokens (e.g., token $i$ vs. token $j$). This bias highlights or de-emphasizes tokens based on how close or far they are in the sequence.

### 1.2.3.3 Advanced Methods

- **Rotary Positional Embedding (RoPE)**: Rotates token embeddings in a multidimensional space based on position, ensuring that the dot product depends only on relative positions. Follow this link to learn more on [RoPE](https:KamilaKare.github.io/edit/main/Formations/LLM%20for%20DS/Module%200/ROPE.md).
- **ALiBi**: Adds a linear bias term to the attention score to represent the distance between tokens.

---

## 1.3 Feed-Forward Layers in the Transformer

After the **attention** sub-layer (whether self-attention or cross-attention), each position in the sequence passes through a **feed-forward** network. This step **increases the model’s capacity** to transform and represent information, operating on each token **independently**.

### 1.3.1 Position-Wise Feed-Forward Network

#### 1.3.1.1 Core Idea

In Transformers, the feed-forward network (FFN) is applied **independently** to each token’s representation. That means for every token $\mathbf{x}_i$, we perform:

$$
\mathbf{z}_i = \text{FFN}(\mathbf{x}_i),
$$

where $\mathbf{z}_i$ is the output of the feed-forward network for token $i$. Since this transformation is the same for every token, it’s called **position-wise**.

#### 1.3.1.2 Architecture

A typical feed-forward network consists of **two linear transformations** with an activation function (e.g., **ReLU** or **GELU**) in between:

$$
\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\,\mathbf{W}_2 + \mathbf{b}_2,
$$

or more explicitly with a named activation:

$$
\mathbf{h} = \text{Activation}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1),
\quad
\mathbf{z} = \mathbf{h}\mathbf{W}_2 + \mathbf{b}_2.
$$

- **$\mathbf{W}_1, \mathbf{b}_1$**: first linear layer parameters
- **$\mathbf{W}_2, \mathbf{b}_2$**: second linear layer parameters
- **$\text{Activation}$**: often **ReLU** or **GELU**

### 1.3.1.3 Dimensionality

- If the embedding dimension is $d_{\text{model}}$, the hidden layer often has a larger dimension (e.g., \(4 \times d_{\text{model}}\)).
- This **expansion** allows the network to learn a richer set of transformations before projecting back to \(d_{\text{model}}\).

### 1.3.2 Why Do We Need a Feed-Forward Network?

1.  **Complements Attention:** The attention mechanism handles **contextual mixing** across tokens, but the feed-forward layer gives **per-token nonlinearity** to expand representational power.
2.  **Per-Token Computation:** Each token is processed independently, preserving **parallelism** and  allowing the model to learn transformations that are not solely dependent on cross-token relationships.
3. **Flexibility**: By varying hidden-layer size, activation, or number of layers, we can tune the network to different tasks and data scales.




### 1.3.3 Integration with Other Sub-Layers

1. **Residual Connections**: As in the rest of the Transformer, the FFN output is **added** to the sub-layer input (residual connection) and then **normalized** (LayerNorm).
2. **Dropout**: Often applied between or after linear transformations to **reduce overfitting**.
3. **Order**: The typical order within a Transformer block is:
   1. Multi-Head Attention + Residual + LayerNorm
   2. **Feed-Forward Network + Residual + LayerNorm**

---

## 1.4 Residual Connections & Layer Normalization

In a Transformer block, **residual connections** and **layer normalization** play a crucial role in stabilizing training and ensuring the flow of gradients. They are used around both the **multi-head attention** sub-layer and the **feed-forward** sub-layer.

---

## 1.4.1 Residual Connections

### 1.4.1.1 Motivation

- **Deep Neural Networks**: As models get deeper, gradients can vanish or explode, making training difficult.
- **Shortcut Paths**: Residual (or skip) connections provide a direct path for gradients, enabling **more stable** and **faster** training.

### 1.4.1.2 Definition

A **residual connection** adds the input of a sub-layer to its output. If $\mathbf{x}$ is the input and $ \text{SubLayer}(\mathbf{x})$ is some transformation (e.g., attention or feed-forward), the output becomes:

$$
\mathbf{x}' = \mathbf{x} + \text{SubLayer}(\mathbf{x})
$$

This helps preserve the **original representation** and lets the sub-layer learn **incremental refinements**.

### 1.4.1.3 Benefits

1. **Eases Optimization**: The sub-layer only needs to learn a “residual” function, often making training more stable.
2. **Better Gradient Flow**: Gradients can bypass the sub-layer if needed, mitigating vanishing/exploding gradients.
3. **Deeper Architectures**: Residual connections enable building deeper networks without losing trainability.


### 1.4.2. Layer Normalization

#### 1.4.2.1 Concept

Unlike **batch normalization**, which normalizes across the batch dimension, **layer normalization** normalizes the features across each sample independently. For an input $\mathbf{h} \in \mathbb{R}^{d}$:

$$
\text{LayerNorm}(\mathbf{h}) = \frac{\mathbf{h} - \mu}{\sigma} \cdot \gamma + \beta
$$

where:
- $\mu$ is the mean of $\mathbf{h}$ across the feature dimension,
- $\sigma$ is the standard deviation across the same dimension,
- $\gamma$ and $\beta$ are learnable parameters for scaling and shifting.

#### 1.4.2.2 Why LayerNorm?

1. **Parallelization**: LayerNorm does not depend on the batch size, making it easier to handle variable batch sizes or sequences.
2. **Stabilized Activations**: By normalizing across features, the model’s activations stay within a manageable range.
3. **Better for NLP**: Transformers often deal with large sequences; LayerNorm helps each token’s representation remain stable, even as attention outputs vary.

### 2.3 Placement in the Transformer

Each sub-layer (multi-head attention or feed-forward) is followed by:
1. A **residual connection** that adds the sub-layer’s output to the input.
2. A **layer normalization** operation:

$$
\mathbf{x}' = \text{LayerNorm}\Bigl(\mathbf{x} + \text{SubLayer}(\mathbf{x})\Bigr)
$$

This pattern is repeated throughout the **encoder** and **decoder** blocks. By chaining these sub-layers with **residual connections** and **layer normalization**, Transformers can train deep architectures effectively.






