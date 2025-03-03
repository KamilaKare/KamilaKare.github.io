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

#### N-Gram and Markov Models

To manage computational complexity, language models often approximate the chain rule by considering only a fixed number of preceding words, known as n-grams. For example, a bigram model (2-gram) simplifies the conditional probability to depend only on the immediately preceding word:
$$
P(y_1, y_2, \ldots, y_n) \;=\; \prod_{i=1}^{n} P(y_i \vert _1, \ldots, y_{i-1})
$$

# Module 2: The Markov Assumption and First-Order Markov Models

### 2.1 The Markov Assumption
- The **Markov condition** states that: "The future is independent of the past, given the present."
  - **For sequence modeling**, this means that we only need the most recent context to predict the next word.
- The key idea is that the prediction at time \( t \) depends only on the previous state, not on the entire history of the sequence.

### 2.2 First-Order Markov Models
- In a **first-order Markov model**, the probability of the next word depends only on the previous word:
  \[
  P(w_n | w_1, w_2, \dots, w_{n-1}) = P(w_n | w_{n-1})
  \]
- **Example**: Given the sequence "I am going to the," the model predicts the next word based on "the" (e.g., "store," "park").

### 2.3 Markov Chains and N-grams
- **N-grams** are a generalization of the first-order Markov model where the prediction depends on the previous \( N \) words.
  - **Bigram (2-grams)**: Depends on the previous word.
  - **Trigram (3-grams)**: Depends on the previous two words.
- Training an N-gram model involves counting the occurrences of word pairs or triplets in a corpus and estimating their probabilities.
- **Limitation**:
  - The model only captures local context, and many word combinations may not appear in the training data (data sparsity).
  - Example: For "I went to the park" → the trigram model would use "to the park" to predict the next word.

---

## Module 3: Limitations of Traditional N-gram Models

### 3.1 Issues with N-grams
- **Data sparsity**: As \( N \) increases, the number of possible word sequences increases exponentially, leading to many unseen word combinations.
- **Limited context**: Even with higher-order N-grams, the model only considers a fixed window of past words (e.g., bigrams, trigrams), which is often insufficient for capturing longer dependencies.
- **Computational complexity**: Storing and processing large N-gram models becomes computationally expensive as \( N \) increases.

### 3.2 Moving Beyond N-grams
- To overcome these limitations, models need to account for more extensive context, beyond just the last few words.
- The **neural network-based models** like RNNs were introduced to address the long-range dependencies that N-grams cannot capture.


#### RNNs and LSTMs

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

