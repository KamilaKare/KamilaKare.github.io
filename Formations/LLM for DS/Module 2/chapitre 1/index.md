---
layout: default
title: Transformer Building Blocks
---

# Chapter 1: Transformer Building Blocks

## 1.1 Queries, Keys, and Values in Attention Mechanisms

### Motivation

Let’s assume we have some data points:
$$
\mathcal{D} = \big\{ (x_1, y_1), (x_2, y_2), \dots, (x_n, y_n) \big\} \in \mathbb{R}^d, \quad d \geq 1
$$
We want to predict $y$ for a new input $x$. 
Given a new input $ x$, how do we estimate $ y $?  
A classic nonparametric regression approach (Nadaraya–Watson) suggests, we can estimate $ y $ using a weighted sum of observed values:

$$
\hat{y} = \sum_{i=1}^{n} \alpha(x, x_i) y_i
$$

where $\alpha$ is  a **kernel function** that determines how similar $x$ is to each $x_i$. The resulting $\hat{y}$ is a weighted average of the observed $y_i$ 
with weights depending on the similarity $\alpha(x, x_i)$. 

### Connecting to Attention

The **attention mechanism** extends this idea. Instead of input-output pairs $ (x_i, y_i) $, we define:

- **Keys $(\mathbf{k}_i)$**: Analogous to the $x_i$ in nonparametric regression, each token has a key vector that represents “where” it might be relevant.
- **Values (\(\mathbf{v}_i\))**: Analogous to $y_i$, each token has a value vector that contains the actual content or information to be retrieved ( content associated with each key).
- **Query (\(\mathbf{q}\))**: Analogous to the new input $x$. It's what we use to figure out which \(\mathbf{k}_i\) are most relevant.

Now, given a **database** $\mathcal{D} = \{(\mathbf{k}_1, \mathbf{v}_1), (\mathbf{k}_2, \mathbf{v}_2), \dots, (\mathbf{k}_m, \mathbf{v}_m)\}$ of key-value pairs. For a given query $\mathbf{q}$,
attention pools the values via weights $ \alpha(\mathbf{q}, \mathbf{k}_i)$

$$
\text{Attention}(\mathbf{q}, \mathcal{D}) = \sum_{i=1}^{m} \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i
$$

where \( \alpha(\mathbf{q}, \mathbf{k}_i) \) are attention weights.

### Attention Pooling

The attention weights \( \alpha(\mathbf{q}, \mathbf{k}_i) \) have specific properties:

- They are **nonnegative**: \( \alpha(\mathbf{q}, \mathbf{k}_i) \geq 0 \), The model never assigns negative importance.
- They form a **convex combination**: \( \sum_{i} \alpha(\mathbf{q}, \mathbf{k}_i) = 1 \).
- If all weights are **equal** (\(\alpha = \frac{1}{m}\)), we’re simply averaging all $\mathbf{v}_i$ equally, which is a naive baseline..

The beauty of attention is that it learns to focus on the relevant keys (and thus their values) based on the query, effectively performing a data-driven weighting akin to nonparametric smoothing.

### Scaling to Large Models

In modern Transformer networks, the function $ \alpha(\mathbf{q}, \mathbf{k}_i) $ is often computed as:

$$
\alpha(\mathbf{q}, \mathbf{k}_i) \propto \exp \left( \frac{\mathbf{q} \cdot \mathbf{k}_i}{\sqrt{d_k}} \right)
$$

followed by a **softmax** operation to normalize the weights. This resembles a **learned kernel function** that adapts based on the training data.

![image](https://github.com/user-attachments/assets/9e9a4a9d-2416-45ea-b49c-1428feffbc2a)

mettre attention avec softmax

This perspective helps us understand why attention mechanisms are so powerful: they **dynamically select the most relevant “neighbors”** in the dataset and **combine their values** based on similarity to the query.


## Attention Mechanism Formula

Given a set of **queries** $ \mathbf{Q} $, **keys** $ \mathbf{K} $, and **values** $ \mathbf{V} $, the attention mechanism computes an output as a weighted sum of values:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax} \left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}
$$

where:

- $ \mathbf{Q} \in \mathbb{R}^{n_q \times d_k} $ is the **query matrix** (representing the search input).
- $ \mathbf{K} \in \mathbb{R}^{n_k \times d_k} $ is the **key matrix** (representing stored information).
- $ \mathbf{V} \in \mathbb{R}^{n_k \times d_v} $ is the **value matrix** (containing data to be retrieved).
- $ d_k $ is the **dimensionality of keys and queries**.
- $ \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} $ computes the **scaled dot-product similarity** between queries and keys.
- The **softmax function** ensures that the attention scores sum to 1, assigning different levels of importance to different values.

### Expanded Form

For a single query $ \mathbf{q} $ and a database $ \mathcal{D} = \{ (\mathbf{k}_1, \mathbf{v}_1), \dots, (\mathbf{k}_m, \mathbf{v}_m) \} $, attention is computed as:

$$
\text{Attention}(\mathbf{q}, \mathcal{D}) = \sum_{i=1}^{m} \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i
$$

where:

$$
\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp \left( \frac{\mathbf{q} \cdot \mathbf{k}_i}{\sqrt{d_k}} \right)}{\sum_{j=1}^{m} \exp \left( \frac{\mathbf{q} \cdot \mathbf{k}_j}{\sqrt{d_k}} \right)}
$$

This defines a probability distribution over the values, focusing more on the most **relevant** ones.

## Multi-Head Attention

### Motivation

In real-world applications, using a single attention mechanism may **limit the model’s ability** to capture diverse relationships within a sequence. A single attention head might focus on **short-range dependencies**, while another might capture **long-range dependencies**. To **enhance the model’s flexibility**, we need a mechanism that can learn **multiple aspects of attention** simultaneously.

To achieve this, instead of applying a **single attention pooling**, we transform the **queries, keys, and values** using **independently learned linear projections**. These projected representations are then **processed by multiple attention mechanisms (heads) in parallel**. Each head specializes in **extracting different types of information** from the input data.

Once all attention heads generate their outputs, these are **concatenated** and passed through another **learned linear transformation**. This final step integrates the information from all heads, producing a richer and more informative representation.

This approach is known as **Multi-Head Attention**, where each individual attention mechanism operates as an independent **head** within the overall framework. By leveraging multiple attention heads, the model gains the ability to focus on different aspects of the input, leading to **improved contextual understanding** and **enhanced learning efficiency**.

### Multi-Head Attention Mechanism

Instead of applying **a single attention operation**, we **project** queries, keys, and values into multiple **subspaces** using independently learned **linear transformations**. Each subspace captures different **aspects** of the relationships between tokens.

1. Each query, key, and value undergoes a learned linear transformation:
   
   \[
   \mathbf{Q}_i = \mathbf{Q} \mathbf{W}_i^Q, \quad 
   \mathbf{K}_i = \mathbf{K} \mathbf{W}_i^K, \quad
   \mathbf{V}_i = \mathbf{V} \mathbf{W}_i^V
   \]

   where \( \mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V \) are the weight matrices for the **i-th attention head**.

2. Each **head** performs **scaled dot-product attention**:

   \[
   \text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i) = \text{softmax} \left( \frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_k}} \right) \mathbf{V}_i
   \]

3. The **outputs of all heads** are concatenated:

   \[
   \text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O
   \]

   where \( \mathbf{W}^O \) is a learned **output projection matrix**.

### Benefits of Multi-Head Attention

- **Captures different types of dependencies** within a sequence.
- **Attends to information at multiple representation subspaces**.
- **Improves model robustness** by allowing different perspectives on data.

Multi-Head Attention is a **core component of the Transformer architecture**, enabling it to process contextual information efficiently. 

### Summary

- The **dot-product attention** scores the similarity between queries and keys.
- The **softmax operation** normalizes these scores into attention weights.
- The **weighted sum of values** gives the output representation.
- **Multi-head attention** enables multiple perspectives on the same input.

This mechanism is the foundation of **Transformers**, enabling powerful architectures like **GPT, BERT, and T5**.

Some intuition might help here: for instance, in a regression setting, the query might correspond to the location where the regression should be carried out. The keys are the locations where past data was observed and the values are the (regression) values themselves.

By design, the attention mechanism provides a differentiable means of control by which a neural network can select elements from a set and to construct an associated weighted sum over representations.
