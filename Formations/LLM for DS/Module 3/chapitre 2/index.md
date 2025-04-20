---
layout: default
title: Advanced Prompt Engineering
---


# Chapter 2: LLM Hyperparameters and Their Influence

Large Language Models (LLMs) rely on various hyperparameters that govern how they generate text. In this chapter, we explore the most important hyperparameters, explain how they affect model behavior, and provide practical guidelines to optimize prompts for improved outputs.

---

## 2.1. Key Hyperparameters Overview

### Temperature

- **Definition**:  
  Temperature controls the randomness or creativity in the model’s predictions. Lower values (e.g., 0.2–0.4) produce more deterministic and focused results, while higher values (e.g., 0.7–1.0) foster creative and diverse outputs.
- **Impact on Prompting**:  
  Use lower temperature settings when you need precise, consistent answers (e.g., technical explanations or legal documents). For creative writing or brainstorming, higher temperatures may yield more varied and imaginative responses.

### Incorporating Temperature into the Softmax Formula

#### The Base Softmax Formula

For a given vector of logits $\mathbf{z} = [z_1, z_2, \dots, z_n]$, the standard softmax function is defined as:

$$
P(i) = \frac{\exp(z_i)}{\sum_{j=1}^{n} \exp(z_j)}
$$

where:
- $P(i)$ is the probability assigned to the $i^{th}$ token.
- $z_i$ is the logit for the $i^{th}$ token.

#### Adding Temperature $T$

The temperature parameter $T$ is introduced to control the randomness of the sampling process. The modified softmax formula with temperature is given by:

$$
P(i) = \frac{\exp\left(\frac{z_i}{T}\right)}{\sum_{j=1}^{n} \exp\left(\frac{z_j}{T}\right)}
$$

##### Effects of Temperature
- **Lower Temperature $(T < 1)$**:  
  Dividing the logits $z_i$ by a value less than one makes the differences between them more pronounced. This causes the softmax output to be **sharper** (i.e., more "confident" or deterministic), as the highest logits get even more dominant.
  
- **Higher Temperature $(T > 1)$**:  
  Dividing the logits $z_i$ by a value greater than one reduces the differences between them. The resulting softmax distribution is **flatter** (i.e., more random or diverse), providing higher probabilities to tokens that might have had lower probabilities with the standard softmax.

- **Special Case $(T = 1)$**:  
  The formula simplifies to the original softmax function.

### Practical Example

Suppose you have logits:
$$
\mathbf{z} = [2.0, 1.0, 0.1]
$$
- **Without temperature adjustment ($T=1$)**:

  $$
  P(i) = \frac{\exp(z_i)}{\exp(2.0) + \exp(1.0) + \exp(0.1)}
  $$
  yielding $P = [0.66, 0.24, 0.1]$

- **With temperature $T = 0.5$**:

  $$
  P(i) = \frac{\exp\left(\frac{z_i}{0.5}\right)}{\exp\left(\frac{2.0}{0.5}\right) + \exp\left(\frac{1.0}{0.5}\right) + \exp\left(\frac{0.1}{0.5}\right)}
  $$

  The exponent values will be higher (i.e., $P = [0.86, 0.12, 0.02]$), resulting in a sharper distribution.

- **With temperature $T = 0.1$**, $P = [0.99, 4.5 * 10^{-5}, 5.6 * * 10^{-9}]$

### Summary

By incorporating the temperature $T$ as:

$$
P(i) = \frac{\exp\left(\frac{z_i}{T}\right)}{\sum_{j=1}^{n} \exp\left(\frac{z_j}{T}\right)}
$$

you can control how “peaked” or “flat” the output distribution is. Adjusting $T$ lets you trade off between more deterministic versus more creative outputs when generating text.


### Top-k and Top-p (Nucleus Sampling)

- **Top-k Sampling**:  
  The model considers only the $k$ most likely tokens for each step. This truncation helps to discard low-probability words.
- **Top-p (Nucleus) Sampling**:  
  Instead of a fixed number of tokens, the model picks tokens from the smallest set whose cumulative probability exceeds a threshold $p$ (e.g., 0.9). This allows more adaptability than top-k.
- **Influence**:  
  These sampling strategies balance between creativity and coherence. More aggressive sampling (lower k or lower p) reduces diversity but increases focus, whereas looser parameters can introduce unexpected or creative outputs.

## Examples of Top-k and Top-p Sampling

### Top-k Sampling Example

Imagine a vocabulary of 5 tokens with the following logits:

- **Token A:** 2.0 
- **Token B:** 1.5  
- **Token C:** 0.5  
- **Token D:** 0.1  
- **Token E:** -1.0  

Using standard softmax, each token's logit is converted to a probability. In top‑k sampling, suppose we set $ k = 3$. The process is as follows:

1. **Identify the Top k Tokens:**  
   Determine which tokens have the highest $ k$ logits. In this example, the top 3 tokens are:
   - Token A (2.0)
   - Token B (1.5)
   - Token C (0.5)

2. **Discard the Rest:**  
   Set the probabilities of tokens not in the top 3 (i.e., Token D and Token E) to 0.

3. **Re-normalize the Probabilities:**  
   Re-calculate the probabilities of the remaining tokens (A, B, and C) so that they sum to 1. We obtain
  - **Token A:** 0.55
  - **Token B:** 0.33  
  - **Token C:** 0.12
   

**Result:** The final probability distribution only includes Tokens A, B, and C, focusing the model's output on the most likely options.

---

### Top-p (Nucleus) Sampling Example

Now, consider the same set of logits, but apply top‑p sampling with a threshold $p$ (for example, $p = 0.8$). The steps are as follows:

1. **Sort Tokens by Probability:**  
   Convert the logits to probabilities using softmax and then rank the tokens in descending order of these probabilities.

2. **Select the Minimal Set:**  
   Starting from the token with the highest probability, add tokens until the cumulative probability reaches at least \( p = 0.8 \).  
   For instance, if after applying softmax you get something like:  
   - Token A: 0.5  
   - Token B: 0.3  
   - Token C: 0.1  
   - Token D: 0.07  
   - Token E: 0.03  
   Adding Token A (0.5) and Token B (0.3) gives a cumulative probability of 0.8, which meets the threshold. (In another scenario, you might need to include Token C if the cumulative sum remains below 0.8.)

3. **Re-normalize over the Selected Tokens:**  
   All tokens that were not included in this minimal set have their probabilities set to 0. The probabilities of the selected tokens are then re-normalized so that their total probability sums to 1.

**Result:** The final output is determined solely by the selected tokens whose combined probability was at least $ p$, ensuring that the model considers only the most influential words as deemed by their cumulative probability.

---

### Summary

- **Top-k Sampling:** Selects a fixed number ($k$) of highest-probability tokens, discards the rest, and re-normalizes the probabilities for the chosen tokens.
- **Top-p Sampling:** Dynamically selects the smallest set of tokens whose cumulative probability is at least $p$ (nucleus), sets the probabilities of other tokens to 0, and then re-normalizes within this set.

Both techniques help control the balance between diversity and determinism in the model's outputs.


### Maximum Tokens and Context Window Size

- **Definition**:  
  The `max_tokens` parameter limits how many tokens the model generates in one call. The context window defines how much previous conversation or text the model can “remember” when generating responses.
- **Considerations**:  
  - Ensure that the prompt plus expected output fits within the model’s maximum context size.  
  - For tasks with multi-step reasoning (like chain-of-thought prompts), the context window must be large enough to accommodate intermediate steps.

### Penalty Parameters

- **Repetition Penalty**:  
  Discourages the model from repeating the same phrases or words, which is particularly useful for lengthy outputs.
- **Frequency and Presence Penalties**:  
  - **Frequency Penalty**: Reduces the chance of repeated words based on how often they appear in the generated text.  
  - **Presence Penalty**: Encourages the inclusion of new and diverse tokens by penalizing tokens that have already appeared.
- **Effect on Outputs**:  
  These penalties help achieve a balance between answer coherence and originality. They are especially useful when crafting prompts that require the model to generate lists or long paragraphs without redundancy.

---

## 2.2. How Hyperparameters Affect Prompting

### Interactive Prompt Optimization

Hyperparameters have a direct impact on the behavior of LLMs. For example:
- A prompt designed for a technical explanation might need low temperature and tight top-p settings to ensure accuracy.  
- In contrast, a creative storytelling prompt might benefit from a higher temperature and a more relaxed top-k sampling to encourage variation.

### Model-Specific Behavior

Different LLM families and versions (e.g., GPT-3, GPT-4, Claude) may respond differently even when the same hyperparameters are applied. It is crucial to:
- Test your prompts with variations in temperature, sampling strategies, and context limits.
- Document these settings to build a repository of best practices for different model architectures.


