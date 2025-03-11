---
layout: default
title: Rotary Positional Embedding (RoPE)
---

# Rotary Positional Embedding (RoPE)

Rotary Positional Embedding (RoPE) is an **innovative approach** to incorporating **relative positional information** into Transformer-based models. Unlike fixed or learned absolute embeddings, RoPE encodes positions by **rotating** token embeddings in a continuous vector space, allowing the **dot product** between two position-encoded tokens to depend only on their **relative positions**.

---

## 1. Motivation

- **Parallel Processing**: Transformers process all tokens simultaneously, losing any inherent notion of sequence order.
- **Relative Positioning**: Many tasks care more about the *distance* between tokens than their absolute indices (e.g., local context).
- **Smooth & Continuous**: RoPE’s rotation-based design elegantly handles long sequences while preserving essential distance information.

---

## 2. Key Idea

RoPE applies a **rotation matrix** to each token’s embedding based on its position. For a token at position \( p \), each pair of embedding dimensions \((x^{(2i)}, x^{(2i+1)})\) is rotated by an angle proportional to \( p \). Formally, if \(\theta\) is a predefined base angle:

\[
\begin{pmatrix}
x^{(2i)}_p \\[6pt]
x^{(2i+1)}_p
\end{pmatrix}
\; \longmapsto \;
\begin{pmatrix}
\cos(p \,\theta_i) & -\sin(p \,\theta_i) \\[3pt]
\sin(p \,\theta_i) & \cos(p \,\theta_i)
\end{pmatrix}
\begin{pmatrix}
x^{(2i)}_p \\[3pt]
x^{(2i+1)}_p
\end{pmatrix},
\]

where \(\theta_i\) is chosen so that **higher embedding dimensions** rotate more slowly than lower ones.

---

## 3. Why It’s Powerful

1. **Relative Position**: The dot product between two RoPE-encoded vectors depends only on their positional difference, making it inherently **relative**.
2. **Seamless Integration**: RoPE can be easily plugged into existing Transformer architectures, requiring only a **lightweight rotation** of embeddings.
3. **Scalable**: By design, RoPE scales to large sequence lengths without the typical constraints of absolute embeddings.

---

## 4. Practical Implementation

1. **Define Angles**: Precompute \(\theta_i\) for each dimension \( i \).  
2. **Rotate Embeddings**: For each token position \( p \), apply the rotation matrix to its embedding slice.  
3. **Feed into Attention**: Use these **rotated embeddings** as queries, keys, and values, preserving relative distance in attention computations.

Here’s a pseudo-code snippet in Pythonic style:

```python
import torch
import math

def rope_embedding(x, dim, base=10000.0):
    """
    x: tensor of shape (batch, seq_len, dim)
    dim: total embedding dimension
    base: base frequency for rotation
    """
    # Split into pairs (2i, 2i+1)
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]

    # positions: range of sequence length
    seq_len = x.shape[1]
    positions = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(1)

    # frequencies for each dimension pair
    idx = torch.arange(dim // 2, dtype=torch.float, device=x.device)
    freqs = base ** (-2.0 * idx / dim)

    # shape them to broadcast
    angles = positions * freqs.unsqueeze(0)

    # compute sin, cos
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    # rotate
    x_even_rot = x_even * cos_vals - x_odd * sin_vals
    x_odd_rot  = x_even * sin_vals + x_odd * cos_vals

    # interleave even/odd
    return torch.stack((x_even_rot, x_odd_rot), dim=-1).flatten(-2)
