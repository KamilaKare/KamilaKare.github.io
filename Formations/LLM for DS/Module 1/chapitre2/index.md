---
layout: default
title: LLM Fundamentals
---

# Chapter 2: LLM Fundamentals

## 2.1  Introduction & Definition
### 2.1.1 Mathematical definitions

#### Reminder: What is a Probability?
A probability distribution is a mathematical function that describes the likelihood of different outcomes in a random experiment. It is a measure between 0 and 1 indicating how likely an event is to occur. 

- **Example 1**: The probability that a fair coin lands on heads is 0.5 (50%).  
- **Example 2**: The probability of rolling a 6 on a fair six-sided die is $\frac{1}{6} \approx 0.167$.  

All possible outcomes in a given scenario sum to 1 when considering their probabilities.

#### Language Model as a Probability Distribution

A **Language Model (LM)**  can be defined as a probability distribution over sequences of words. It assigns a probability to every possible sequence of words $(y_1, y_2, \ldots, y_n)$. Formally:

$$
\mathbb{P} (y_1, y_2, \ldots, y_n).
$$

- **Interpretation**:
  - A higher $\mathbb{P}(\dots)$ means the model deems that sequence more likely or more natural in a given language.  
  - For text generation, the LM can sample from this distribution to produce new sequences.

---

#### Chain Rule 
In language modeling, the chain rule of probability allows us to decompose the joint probability of a sequence of words into a product of conditional probabilities. This decomposition is expressed as:

$$
P(y_1, y_2, \ldots, y_n) \;=\; P(y_1)\prod_{i=2}^{n} P(y_i \vert _1, \ldots, y_{i-1})
$$

Here, each word $y_i$ is conditioned on all preceding words $y_1$ to $y_{i-1}$, capturing the context up to that point. This approach enables the model to consider the entire prior context when predicting the next word, making it **auto-regressive**.

#### Auto-Regressive Models
Auto-regressive models generate each word in a sequence based on the previously generated words. By leveraging the chain rule, these models can produce coherent and contextually relevant text. However, modeling long sequences can be computationally intensive due to the dependency on all prior words.

### 2.1.2 Evolution of Language Models

#### The Markov Assumption and First-Order Markov Models
To manage computational complexity, language models often approximate the chain rule by considering only a fixed number of preceding words.

#### The Markov Assumption
- The **Markov condition** states that: "The future is independent of the past, given the present."
  - **For sequence modeling**, this means that we only need the most recent context to predict the next word.
- The key idea is that the prediction at time $t$ depends only on the previous state, not on the entire history of the sequence.

#### First-Order Markov Models
- In a **first-order Markov model**, the probability of the next word depends only on the previous word:
$$ P(y_n \vert y_1, y_2, \dots, y_{n-1}) = P(y_n \vert y_{n-1}) $$

- **Example**: Given the sequence "I am going to the," the model predicts the next word based on "the" (e.g., "store," "park").

#### Markov Chains and N-grams
- **N-grams** are a generalization of the first-order Markov model where the prediction depends on the previous $N$ words.
  - **Bigram (2-grams)**: Depends on the previous word.
  - **Trigram (3-grams)**: Depends on the previous two words.
- Training an N-gram model involves counting the occurrences of word pairs or triplets in a corpus and estimating their probabilities.  In other words, in order to compute the language model, we need to calculate the probability of words and the conditional probability of a word given the previous few words. Note that such probabilities are language model parameters.

#### Issues with N-grams
- **Data sparsity**: As $N$ increases, the number of possible word sequences increases exponentially, leading to many unseen word combinations.
- **Limited context**:  N-gram models consider only a fixed window of preceding words (e.g., bigrams, trigrams). This fixed context window is often insufficient for capturing longer-range dependencies and nuances in language, limiting the model's ability to understand and generate coherent text over longer passages.
- **Computational complexity**: Storing and processing large N-gram models becomes computationally expensive as $N$ increases. This complexity can slow down processing and make the models less practical for large-scale applications.
- **Out-of-Vocabulary (OOV) Words**: N-gram models struggle with words or sequences that were not present in the training data. When encountering these OOV words during testing or real-world application, the model may fail to provide accurate predictions or generate meaningful text.

#### Moving Beyond N-grams
- To overcome these limitations, models need to account for more extensive context, beyond just the last few words.
- The **neural network-based models** like RNNs were introduced to address the long-range dependencies that N-grams cannot capture.

---
#### RNNs and LSTMs

- **Neural networks** can be used to model sequences of words by learning patterns in the data.
- **Feedforward neural networks** were an early attempt at sequence modeling, but they lack the capability to capture sequential dependencies.

#### Recurrent Neural Netowrks (RNNs)
- **Architecture**:
    ![image](https://github.com/user-attachments/assets/651f0ee9-3f0f-4c57-b5ec-20dbf87f722c)
  <figure>
  ![image](https://github.com/user-attachments/assets/651f0ee9-3f0f-4c57-b5ec-20dbf87f722c)
  <figcaption>RNN Architecture.</figcaption>
</figure>

  - RNNs have loops in their architecture that allow information to persist across time steps.
  - At each time step $t$, the network computes a hidden state  $h_t$ using:
    $$
    h_t = f(W_x x_t + W_h h_{t-1} + b)
    $$
    where:
    - $x_t$ is the input at time $t$,
    - $h_{t-1}$ is the hidden state from the previous time step,
    - $W_x$ and $W_h$ are weight matrices,
    - $b$ is a bias term,
    - $f$ is an activation function (typically $\tanh$ or ReLU).
      
- **Challenges**:
  - **Vanishing gradients**: During training, gradients may shrink exponentially over long sequences, making it difficult to learn long-range dependencies.
  - **Exploding gradients**: In some cases, gradients can grow uncontrollably, destabilizing training.
  - **Sequential processing**: RNNs process one time step at a time, limiting parallelization.

#### Long Short-Term Memory (LSTM) Networks

- **Purpose**: LSTMs are designed to overcome the limitations of standard RNNs, particularly the vanishing gradient problem.
- **Core Component – The Memory Cell**:
  - Each LSTM cell contains a **cell state** $C_t$ that serves as a memory, allowing information to flow relatively unchanged over time.
- **Gating Mechanisms**:
  - **Forget Gate** $f_t$:
    
    $$
    f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
    $$
    Decides what information to discard from the cell state.
  - **Input Gate** $i_t$:
    
    $$
    i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
    $$
    
    Determines which new information to add to the cell state.
  - **Candidate Cell State** \( \tilde{C}_t \):
    \[
    \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
    \]
    Represents the new candidate values to be added.
  - **Output Gate** \( o_t \):
    \[
    o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
    \]
    Decides what part of the cell state to output.
- **Updating the Cell State and Hidden State**:
  - The new cell state is updated as:
    \[
    C_t = f_t \ast C_{t-1} + i_t \ast \tilde{C}_t
    \]
  - The hidden state is then computed by:
    \[
    h_t = o_t \ast \tanh(C_t)
    \]
- **Benefits of LSTMs**:
  - They can maintain and update long-term dependencies much more effectively than standard RNNs.
  - LSTMs mitigate the vanishing gradient problem, allowing training on longer sequences.
  - Their gating mechanisms provide a way to selectively remember or forget information, which is crucial for tasks requiring context over long distances.

---



#### LLMs


## 2.2 Core Concepts

### 2.2.1 Tokenization

### 2.2.2 Embeddings

### 2.2.3 Attention Mechanism Overview

### 2.2.4 Popular Architectures

## 2.3 Why LLMs Matter

### 2.3.1 Handling context at Scale

### 2.3.2 Transfer learning advantage

### Real-time Adaptation 

