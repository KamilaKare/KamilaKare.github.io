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
P(y_1, y_2, \ldots, y_n) \;=\; \prod_{i=1}^{n} P(y_i \vert _1, \ldots, y_{i-1})
$$

Here, each word $y_i$ is conditioned on all preceding words $y_1$ to $y_{i-1}$, capturing the context up to that point. This approach enables the model to consider the entire prior context when predicting the next word, making it **auto-regressive**.

#### Auto-Regressive Models
Auto-regressive models generate each word in a sequence based on the previously generated words. By leveraging the chain rule, these models can produce coherent and contextually relevant text. However, modeling long sequences can be computationally intensive due to the dependency on all prior words.

### 2.1.2 Evolution of Language Model


## 2.2 Core Concepts

### 2.2.1 Tokenization

### 2.2.2 Embeddings

### 2.2.3 Attention Mechanism Overview

### 2.2.4 Popular Architectures

## 2.3 Why LLMs Matter

### 2.3.1 Handling context at Scale

### 2.3.2 Transfer learning advantage

### Real-time Adaptation 

