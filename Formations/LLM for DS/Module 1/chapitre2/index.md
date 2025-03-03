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
\mathcal{P} (y_1, y_2, \ldots, y_n).
$$

- **Interpretation**: A higher \(P(\dots)\) means the model deems that sequence more likely or more natural in a given language.  
- For text generation, the LM can sample from this distribution to produce new sequences.

---

#### Chain Rule (Optional Detail)
Often, language models factorize this joint probability using the **chain rule**:

\[
P(t_1, t_2, \ldots, t_n) \;=\; \prod_{i=1}^{n} P(t_i \mid t_1, \ldots, t_{i-1}).
\]

Each token depends on the tokens before it, capturing the context.

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

