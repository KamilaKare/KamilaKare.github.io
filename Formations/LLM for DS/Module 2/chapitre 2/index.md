---
layout: default
title: Transformer Architecture
---

# Chapter: Transformer Architecture

With the key **building blocks** (multi-head attention, feed-forward networks, positional encoding, and residual connections + layer normalization) now in hand, we can explore how they come together to form the **Transformer**. Introduced by *Vaswani et al. (2017)*, the Transformer has become a cornerstone of modern deep learning for **sequence-to-sequence** tasks in NLP, speech, and beyond.

---

## 1. Model Overview

A **Transformer** processes sequences in parallel, using **attention** mechanisms instead of recurrence or convolution to capture dependencies. It typically consists of:

1. **Encoder**: A stack of identical layers that encodes the source sequence into a set of continuous representations.  
2. **Decoder**: Another stack of identical layers that decodes those representations into an output sequence, often one token at a time.

Each **encoder** and **decoder** layer includes:
- **Multi-Head Attention** (self-attention in the encoder, self-attention + cross-attention in the decoder),
- **Feed-Forward** sub-layer,
- **Residual Connections** and **Layer Normalization** around each sub-layer.

Additionally, **positional encoding** is added to each token’s embedding to preserve sequence order.

{% include image.html src="https://github.com/user-attachments/assets/9e9a4a9d-2416-45ea-b49c-1428feffbc2a" alt="The Transformer Architecture" caption="The Transformer Architecture." %}

---

## 2. The Encoder

### 2.1 Encoder Architecture

An **encoder** is composed of \(N\) identical layers, where each layer contains two main sub-layers:

1. **Self-Attention Sub-Layer**  
   - The tokens in the **same sequence** attend to each other, determining the context for each position.  
   - Residual connection and layer normalization wrap around this sub-layer.

2. **Feed-Forward Sub-Layer**  
   - A position-wise feed-forward network (often a 2-layer MLP with ReLU or GELU).  
   - Another residual connection and layer normalization follow this sub-layer.

**Conceptual Pseudocode**:

```python
for layer in encoder_layers:
    # 1) Self-Attention
    x = layer_norm(x + self_attention(x, x, x))

    # 2) Feed-Forward
    x = layer_norm(x + feed_forward(x))

## 2 The Decoder Architecture
A decoder also consists of $𝑁$ identical layers, but each layer has three sub-layers:

### 2.1 Masked Self-Attention

The decoder attends to its own partial output sequence, but uses a mask to avoid looking at future tokens.
Followed by a residual connection and layer normalization.

### 2.2 Cross-Attention (a.k.a. Encoder-Decoder Attention)

The decoder attends to the encoder’s output representations.
Another residual connection and layer normalization.

### 2.3 Feed-Forward Sub-Layer

Same position-wise feed-forward network as in the encoder.
Again, residual + layer normalization.

### 3.1 End-to-End Flow

#### **Encoder:**
- **Input tokens**: \( x_1, \dots, x_m \)
- **Embedding + positional encoding**  
  → Pass through \( N \) encoder layers  
  → Produces **encoder outputs** \( h \).

#### **Decoder:**
- **Partial target tokens**: \( y_{<t} \)
- **Embedding + positional encoding**  
  → Pass through \( N \) decoder layers, using both:  
  - **Masked self-attention** on \( y_{<t} \)  
  - **Cross-attention** to \( h \)  
  → Outputs a **hidden representation** \( z_t \).

#### **Projection + Softmax:**
- \( z_t \) is projected to **logits** for each vocabulary token.
- **Softmax** produces a **probability distribution** over the next possible token.

