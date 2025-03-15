---
layout: default
title: Training Transformers 
---


# Chapter 3: Training Transformers  

Training a Transformer model effectively requires careful consideration of loss functions, optimization strategies, and regularization techniques to ensure stability and generalization.  

## 3.1 **Loss Functions**  
The choice of loss function depends on the specific Transformer application.  

### 3.1.1 Cross-Entropy Loss

In **Transformer**-based language models, **cross-entropy** is the standard loss function used for **next-token prediction** and other classification tasks. It measures how well the model’s predicted probability distribution aligns with the true (target) distribution over the vocabulary.

#### 3.1.1.1. Definition

For a vocabulary of size  $V$ and a single training example where the true label is $y$ (one-hot encoded) and the predicted probabilities are  $\hat{y}$, **cross-entropy** is defined as:

$$
\mathcal{L}_{\text{CE}} = - \sum_{i=1}^{V} y_i \log(\hat{y}_i).
$$

- $y_i$ is 1 if $ i$ is the correct class (token), otherwise 0.  
- $\hat{y}_i$ is the predicted probability for class $i$.  

In practice, for a **next-token prediction** scenario, $y_i$ is 1 for the correct next token index and 0 for all others.


#### 3.1.1.2. Example: Next-Token Prediction

Consider a small vocabulary of size $V = 5$, and a model predicts probabilities for the next token as follows:

| Token Index | Token | Model Prediction $\hat{y}_i$ |
|-------------|-------|---------------------------------|
| 0           | "cat" | 0.20                            |
| 1           | "dog" | 0.50                            |
| 2           | "the" | 0.10                            |
| 3           | "sat" | 0.10                            |
| 4           | "on"  | 0.10                            |

Suppose the **true token** is “dog” (index 1). Then $ y = [0, 1, 0, 0, 0]$.  

**Cross-entropy** for this single prediction is:

$$
\mathcal{L}_{\text{CE}}
= -\sum_{i=0}^{4} y_i \log(\hat{y}_i)
= -\bigl(0 \cdot \log(0.20) + 1 \cdot \log(0.50) + 0 \cdot \log(0.10) + 0 \cdot \log(0.10) + 0 \cdot \log(0.10)\bigr)
= -\log(0.50).
$$

Cross-entropy loss effectively penalizes incorrect predictions. In the ideal scenario where $\hat{y}_1 \approx 1$ for “dog”,  the cross-entropy loss approaches zero, reflecting a correct prediction. Conversely, $\hat{y}_1 \approx 0$ would incur a large loss, signifying a substantial deviation from the target.

#### 3. Cross-Entropy in Language Modeling

In a **language modeling** setting (e.g., GPT-like decoder-only Transformers), we often sum or average cross-entropy across all tokens in a sequence. For a sequence of length $N$:

$$
\mathcal{L}_{\text{LM}} = -\frac{1}{N} \sum_{t=1}^{N} \sum_{i=1}^{V} y_{t,i} \log(\hat{y}_{t,i}),
$$

where $ y_{t,i}$ is the true token at position $ t$, and $ \hat{y}_{t,i}$ is the predicted probability for token $i$ at position 
 $t$.

#### 4. Implementation Tips

1. **Softmax & Logarithm**:  
   - In many frameworks (e.g., PyTorch, TensorFlow), you’ll find functions like `CrossEntropyLoss` or `sparse_categorical_crossentropy` that combine **softmax** and **log** steps for numerical stability.  
2. **Batch-Wise Computation**:  
   **In practice**, we usually process **batches** of sequences for computational efficiency. Let $B$ be the **batch size**, and let each sequence in the batch have length $N$. The loss then becomes:

$$
\mathcal{L}_{\text{LM}} = -\frac{1}{B \times N}
\sum_{b=1}^{B} \sum_{t=1}^{N} \sum_{i=1}^{V} y_{b,t,i} \log\bigl(\hat{y}_{b,t,i}\bigr),
$$

where:
- $ y_{b,t,i}  = 1$ if token $i$ is correct at position $t$ in the $b$-th sequence, and 0 otherwise.
- $\hat{y}_{b,t,i}$ is the predicted probability for token $i$ at position $t$ in the $b$-th sequence.

By **averaging** over both **batch** and **sequence length**, we obtain a single scalar loss, which we then **backpropagate** to update the model’s parameters. This approach significantly speeds up training and utilizes GPU/TPU resources more effectively.

3. **Label Smoothing**:  
  - A regularization technique that prevents overconfidence in predictions.  
  - Instead of assigning a probability of 1 to the correct token and 0 to others, the target distribution is slightly smoothed:

$$
y' = (1 - \epsilon) y + \frac{\epsilon}{V}
$$

where $ \epsilon$ is a small smoothing factor and $ V$ is the vocabulary size. 

#### 5. Why Cross-Entropy?

- **Information-Theoretic Justification**: Cross-entropy measures the distance between two distributions (true vs. predicted).  
- **Differentiability**: Smooth and well-defined gradients for all classes.  
- **Simplicity & Effectiveness**: Works well in practice for classification and language modeling tasks.

**Summary**: Cross-entropy loss remains the **go-to** objective for **Transformer** training, especially in language modeling. By penalizing the model proportionally to the negative log-likelihood of the correct token, it provides a **direct** measure of how well the model predicts the **true** sequence.


## 3.2 **Optimization**  
Training Transformers involves large-scale optimization challenges. The following techniques are commonly used:  

- **AdamW Optimizer**:  
  - A variant of Adam with weight decay to improve generalization.  
  - Helps mitigate overfitting by preventing parameter explosion.  

- **Learning Rate Scheduling**:  
  - Transformers require dynamic learning rates to stabilize training.  
  - A common approach follows the "warmup" strategy:

$$
\text{lr}(t) = d_{\text{model}}^{-0.5} \times \min(t^{-0.5}, t \times \text{warmup}^{-1.5})
$$

where $ d_{\text{model}}$ is the hidden size, $t$ is the current step, and $\text{warmup}$ is the warmup step count.  

### **Regularization**  
To enhance generalization and prevent overfitting, various regularization techniques are applied:  

- **Dropout**:  
  - Applied to attention weights and feed-forward layers.  
  - Prevents neurons from co-adapting and improves robustness.  

- **Weight Decay**:  
  - Adds an $L_2$ penalty to model weights, preventing excessive weight growth.  

- **Gradient Clipping**:  
  - Limits the magnitude of gradients to avoid instability in training.  

Training Transformers efficiently requires a combination of these techniques to ensure smooth convergence, prevent overfitting, and enhance generalization across different NLP tasks.  

### Optimization with the Cross-Entropy Loss

Once the **cross-entropy** (averaged over the batch and sequence) is computed, **optimization** typically follows these steps:

1. **Forward Pass**  
   - The model processes the input batch (e.g., a set of token sequences) to produce **logits** over the vocabulary.

2. **Loss Computation**  
   - A **softmax** layer converts logits to probabilities.
   - The **cross-entropy** is calculated by comparing predicted probabilities $\hat{y}_{b,t,i}$ with the one-hot targets $y_{b,t,i}$.

3. **Backward Pass**  
   - Automatic differentiation frameworks (e.g., PyTorch, TensorFlow) compute gradients of the loss $\mathcal{L}_{\text{LM}}$ with respect to all model parameters.

4. **Parameter Update**  
   - An **optimizer** (e.g., AdamW) uses these gradients to adjust the model’s weights.  
   - Commonly, a **learning rate scheduler** is also employed, starting with a warmup phase and then decaying the learning rate over time.

By **iterating** this forward-backward-update loop across many batches and epochs, the model **minimizes** cross-entropy, thereby **maximizing** the likelihood of the correct tokens and improving language modeling performance.


### Parameter Updates: Gradient Descent & AdamW

Once gradients of the cross-entropy loss are computed (via backpropagation), **parameter updates** follow a gradient-based rule. Two common approaches are **Stochastic Gradient Descent (SGD)** and **AdamW**.

#### 1. Stochastic Gradient Descent (SGD)

In its simplest form, **SGD** updates each parameter $\theta$ by moving it **opposite** to the gradient direction:

$$
\theta \leftarrow \theta \;-\; \eta \,\nabla_{\theta} \,\mathcal{L}{\text{LM}}(\theta),
$$

where:
- $\theta$ represents a **model parameter** (e.g., weights in a Transformer layer).
- $\eta$ is the **learning rate**, controlling the step size.
- $\nabla_{\theta} \,\mathcal{L}{\text{LM}}(\theta)$ is the gradient of the loss $\mathcal{L}$ with respect to $\theta$.

In **mini-batch** training, $\nabla_{\theta}\,\mathcal{L}{\text{LM}}$ is averaged over a small batch of examples rather than the entire dataset.

#### 2. Adam (Adaptive Moment Estimation)

While plain SGD works, modern large-scale models often use **Adam** or **AdamW**, which incorporate **momentum** and **adaptive learning rates** for each parameter. Adam maintains **exponential moving averages** of both gradients and their squares:

1. **Momentum** update (first moment):

$$
m_t \;\leftarrow\; \beta_1\,m_{t-1} \;+\; (1-\beta_1)\,\nabla_{\theta}\,\mathcal{L}{\text{LM}}(\theta)
$$

2. **Variance** update (second moment):

$$
v_t \;\leftarrow\; \beta_2\,v_{t-1} \;+\; (1-\beta_2)\,\bigl(\nabla_{\theta}\,\mathcal{L}{\text{LM}}(\theta)\bigr)^2
$$

3. **Bias-corrected estimates**:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, 
\quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

4. **Parameter update**:

$$
\theta \;\leftarrow\; 
\theta \;-\; \eta \,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

where:
- $$\beta_1,\beta_2$$ are hyperparameters (often 0.9 and 0.999).
- $\epsilon$ is a small constant (e.g., \(10^{-8}\)) for numerical stability.
- $\eta$ is the learning rate.

**Adam** adjusts the learning rate based on how frequently a parameter has been updated, speeding convergence and often requiring less manual tuning.

#### 3. AdamW (Adam + Weight Decay)

**AdamW** is a popular variant that **decouples weight decay** (L2 regularization) from the gradient-based update:

$$
\theta \;\leftarrow\; 
\theta \;-\; \eta\,\Bigl(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \,\theta\Bigr),
$$

where $\lambda$ is a **weight decay** factor. This ensures weight decay is applied directly to $\theta$ rather than mixing it into the Adam moments, generally leading to better generalization.

### Practical Tips

1. **Learning Rate Schedules**  
   - Warmup, then decay (linear or inverse square root) is common in Transformers.
2. **Batch Size**  
   - Larger batch sizes can speed up training but may require scaling the learning rate.
3. **Gradient Clipping**  
   - Prevents exploding gradients by limiting their norm (e.g., $\|\nabla\| \leq 1$).

Combining **cross-entropy loss** with **AdamW** (and possibly **learning rate scheduling**) is the de-facto approach for training large-scale Transformers, balancing **stable convergence** and **robust generalization**.

