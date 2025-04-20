---
layout: default
title: Advanced Prompt Engineering
---


# Chapter 4: Advanced Prompting Techniques

Advanced prompting techniques extend basic zero‑, one‑, and few‑shot methods to enable complex reasoning, domain grounding, and automated optimization. Techniques like Chain‑of‑Thought and ReAct guide models to articulate intermediate reasoning steps, improving performance on multi‑step tasks (Jason Wei et al).
Retrieval‑Augmented Generation (RAG) grounds outputs in external knowledge, mitigating hallucinations 
Automated methods such as the Automatic Prompt Engineer (APE) use LLMs themselves to generate and refine prompts, reducing human effort while maintaining high task performance (Yongchao Zhou et al.)
.Emerging patterns like Tree of Thoughts enable strategic search over reasoning paths, dramatically boosting problem‑solving success rates (Shunyu Yao et al.)
.Together, these techniques form a robust toolkit for data scientists and technical practitioners to push LLM capabilities further.

## 4.1 Zero‑Shot Prompting

Zero-shot prompting involves instructing a language model to perform a task without providing any examples or demonstrations. The model relies solely on its pre-trained knowledge and understanding of the task based on the prompt's wording.

Modern large language models (LLMs), such as GPT-3.5 Turbo, GPT-4, and Claude 3, have been trained on extensive datasets and fine-tuned to follow instructions. This training enables them to handle a variety of tasks in a zero-shot manner, interpreting prompts and generating appropriate responses without needing task-specific examples.

#### Example: Prompt Zero‑Shot:
```
Translate the following English text into French.
Text: I like teaching maths.
Translation: 
```
#### Output
```
J'aime enseigner les mathématiques. 
```

In this example, the model correctly translates the sentence without being provided with any prior examples of translations. This demonstrates the model's ability to understand and execute the task based on the prompt alone.

### 4.1.1 Benefits

- **Efficiency**: Eliminates the need for crafting and including example inputs, saving time and reducing prompt length.

- **Flexibility**: Allows for quick adaptation to various tasks without requiring task-specific data.

- **Simplicity**: Straightforward to implement, making it accessible for users with varying levels of expertise.

### 4.1.2 Limitations
- **Performance Variability**: May underperform on complex or nuanced tasks where examples could provide necessary context.

- **Ambiguity Handling**: Without examples, the model might misinterpret ambiguous prompts or fail to grasp the desired output format.

- **Dependence on Prompt Clarity**: The success of zero-shot prompting heavily relies on the clarity and specificity of the prompt.

### 4.1.3 Enhancements
Research has shown that techniques like instruction tuning and reinforcement learning from human feedback (RLHF) can significantly improve a model's zero-shot performance. Instruction tuning involves fine-tuning models on datasets described via instructions, enhancing their ability to follow prompts accurately. RLHF further aligns models with human preferences, leading to more reliable and contextually appropriate responses.

### 4.2 Few‑Shot Prompting
#### 4.2.1 Overview

Few-shot prompting is a technique where a model is provided with a small number of examples (typically 1–10) in the prompt itself. These examples serve as demonstrations of the task, helping the model understand the pattern or structure it should follow. This technique enables in-context learning, where the model learns from the context of the prompt without requiring additional training.

Few-shot prompting became viable as models were scaled to larger sizes (Kaplan et al., 2020; Brown et al., 2020) and is particularly useful when zero-shot performance is insufficient for complex or unfamiliar tasks.

Let's demonstrate few-shot prompting via an example that was presented in Brown et al. 2020. In the example, the task is to correctly use a new word in a sentence.

**Prompt**:
```
A "whatpu" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is:
We were traveling in Africa and we saw these very cute whatpus.
 
To do a "farduddle" means to jump up and down really fast. An example of a sentence that uses the word farduddle is:
```
**Output**:
```
When we won the game, we all started to farduddle in celebration.
```

We observe that the model is capable of learning how to perform the task with just a single example (i.e., 1-shot learning). For more challenging tasks, performance can often be improved by increasing the number of demonstrations (e.g., 3-shot, 5-shot, 10-shot, etc.).

Based on the findings of Min et al. (2022), here are several key insights about using demonstrations (or exemplars) in few-shot learning:

- Both the label space and the distribution of the input text provided in the demonstrations are crucial, even if the labels themselves are not correct for each example.

- The formatting of demonstrations significantly affects performance. Surprisingly, even using randomly assigned labels is more effective than providing no labels at all.

- Using random labels sampled from the true label distribution (rather than from a uniform distribution) has been shown to further enhance results.

Let’s explore this in practice. We’ll begin by trying an example where the labels "Positive" and "Negative" are randomly assigned to the inputs:

**Prompt**
```
This is fantastic! // Negative  
This is terrible!  // Positive  
That movie was amazing! // Positive  
What an unpleasant day! // ?
```
**Output**
```
Negative
```
We still get the correct answer, even though the labels have been randomized.

#### 4.2.2 Benefits
- **Improved Performance**: Particularly on tasks that are too nuanced or specific for zero-shot methods.

- **No Fine-Tuning Needed**: Achieves task adaptation through in-prompt examples, avoiding the overhead of model retraining.

- **Human-Like Reasoning**: Mimics how humans generalize patterns from a few examples.

#### 4.2.3 Limitations
- **Prompt Length Constraints**: The number of examples is limited by the model's context window.

- **Example Sensitivity**: Model performance is sensitive to the format, quality, and order of examples.

- **Inconsistency**: Small changes in the prompt or example set can lead to large variations in output.


#### 4.2.4 Tips for Effective Few-Shot Prompts
Research by Min et al. (2022) highlights that:

- The **label space** and **distribution of input text** in examples significantly influence outcomes —even if the labels are incorrect.

- **Prompt formatting** is critical. Consistent formatting improves performance, even when labels are randomized.

- Sampling **labels from a realistic distribution** (rather than uniformly) also helps.




## 4.3 Chain‑of‑Thought (CoT) Prompting
### 4.3.1 Overview
Chain‑of‑Thought prompting  elicits intermediate reasoning steps from an LLM before it delivers a final answer, enabling it to tackle complex, multi‑step tasks more accurately. First introduced by Wei et al. (2022), CoT is an **emergent ability** that appears in sufficiently large models and can be applied in both few‑shot and zero‑shot modes.


{% include image.html src="https://github.com/user-attachments/assets/637a767d-b352-4a6f-b1f2-df27d0516375" alt="Wei et al., 2022" caption="Wei et al., 2022" %}

**Prompt**
```
The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.
A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.

The odd numbers in this group add up to an even number: 17,  10, 19, 4, 8, 12, 24.
A: Adding all the odd numbers (17, 19) gives 36. The answer is True.

The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. 
A:
```
**Output**
```
Adding all the odd numbers (15, 5, 13, 7, 1) gives 41. The answer is False.
```

Anoter approach of CoT is the idea of zero-shot CoT (Kojima et al. 2022) that essentially involves adding "Let's think step by step" to the original prompt.

{% include image.html src="https://github.com/user-attachments/assets/797cbc84-3fd5-47c0-b29a-6ff9f02cff9a" alt="Kojima et al., 2022" caption="Kojima et al., 2022" %}

**Prompt**
```
I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. How many apples did I remain with?
```
**Output**
```
10 apples
```
Just a few months ago, chatgpt gave a result of 11. Today, the GPT-4o that powers Chapgpt is much more powerful and integrates CoT. You had to add "Let's think step by step" to get the right answer

**Prompt**
```
I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. How many apples did I remain with?

Let's think step by step.
```
Let's break it down step by step:

1.  You started with 10 apples.
2.  You gave away 2 + 2 = 4 apples.
3.  After giving some away, you had 10 - 4 = 6 apples.
4.  You then bought 5 more apples, so you had 6 + 5 = 11 apples.
5.  Finally, you ate 1 apple, leaving you with 11 - 1 = 10 apples.

So, you remained with **10** apples.
```

### 4.3.2 Benefits
- **Improved Accuracy**: CoT dramatically boosts performance on reasoning benchmarks such as GSM8K and MultiArith .

- **Transparency**: Reveals intermediate steps, aiding debugging and trust .

- **General Applicability**: Effective across domains—math problems, logical deduction, reading comprehension, and spatial reasoning (e.g., gpt‑4o’s accuracy on simple route planning jumps from 12.4% to 87.5% with CoT, Xu and al., 2024)

### 4.3.3 Limitations

- **Token Overhead**: Multi‑step chains consume significant context, risking window overflow on long tasks .

- **Error Propagation**: A mistake in an early step may derail the entire solution .

- **Model Size Dependency**: Smaller models often fail to generate coherent chains; CoT is most reliable in large‑scale LLMs .


## 4.4 Self‑Consistency Decoding
### 4.4.1 Overview
### Definition  
Self‑Consistency Decoding, introduced by Wang et al., 2022 is a technique that samples **multiple** Chain‑of‑Thought (CoT) reasoning paths for the same query and then **selects** the final answer that appears most frequently across those paths .

{% include image.html src="https://github.com/user-attachments/assets/9eb73561-eca8-4663-bb33-b4a146a72661" alt="Wang al., 2022" caption="Wang al., 2022" %}



### Goal  
The goal of self‑consistency is to **stabilize** the model’s output by aggregating across diverse reasoning chains, thereby **mitigating** the chance that an individual, possibly flawed, chain-of-thought leads to an incorrect answer.

### Method  
1. **Generate** \(N\) CoT outputs for the same prompt (e.g., “Let’s think step by step…”).  
2. **Extract** the final answer from each reasoning chain.  
3. **Count** the frequency of each unique answer among the \(N\) samples.  
4. **Select** the answer with the highest vote as the **consensus** output.

Let us illustrate that.
  Q: What is 17 × 12?
- Generate 10 CoT reasoning samples.  
- Possible answers from chains: {204, 200, 204, 204, 204, 200, 204, 204, 204, 204}.  
- The answer “204” appears 8 times; “200” appears twice.  
- **Return:** 204 (the most consistent answer).

### 4.4.2 Benefits  
- **Higher Accuracy**: On benchmarks such as GSM8K, self‑consistency yields significant gains (e.g., +17.9% over single-chain CoT) .  
- **Robustness**: Reduces sensitivity to sampling noise and early-step errors in individual chains.  
- **Interpretability**: Voting statistics can highlight when the model’s answers are uncertain (e.g., close vote counts).

### Limitations  
- **Compute Intensive**: Requires \(N\) forward passes per query, increasing latency and cost.  
- **Diminishing Returns**: Beyond a certain \(N\), additional samples yield smaller accuracy improvements.  
- **Prompt Overhead**: Each chain-of-thought prompt must fit within the model’s context window, limiting complexity when \(N\) is large.

---

## 4.5 Prompt Chaining
### 4.5.1 Overview

Prompt chaining breaks a complex task into subtasks and feeds the output of each prompt as the input to the next, forming a “chain” of operations. By modularizing the workflow, you can isolate errors, refine individual components, and leverage specialized prompts for each subtask.

**Use Cases**
- **Document Question Answering**
    - Step 1: Extract relevant quotes or passages from a long document based on a user’s question.
    - Step 2: Synthesize those quotes into a coherent answer, ensuring accuracy and conciseness 

- **Conversational Agents**
    - Stage 1: Identify user intent (e.g., booking a flight).
    - Stage 2: Gather required details (dates, destinations).
    - Stage 3: Confirm and execute the booking process.
    This structured approach improves user experience by guiding the conversation through logical, manageable steps 

### Example
```
1. Prompt 1 (Extraction)
   You are an assistant. Extract all sentences relevant to the question below from the document delimited by ####.  
####  
{{document}}  
####  
Respond with each quote wrapped in <quote> tags. 
```
Output
```
<quote>The quick brown fox jumps...</quote>
<quote>The fox is often used to illustrate...</quote>
```
**Prompt 2**
```
Using the extracted quotes and the original document (####…####), answer the user’s question in a friendly, concise paragraph.  
```
**Output**
```
The document explains how the quick brown fox is frequently used in typographical...
```
This two‑step chain produces more accurate and contextually rich answers than a single prompt tackling both tasks. 

### 4.5.2 Benefits
- **Enhanced Accuracy**: Specialized prompts for subtasks yield higher overall task performance 
- **Transparency & Debuggability**: Intermediate outputs reveal where errors occur, simplifying troubleshooting 
- **Modularity & Reusability**: Individual chain components can be reused across applications or combined in new sequences 


### 4.5.3 Limitations
- **Error Propagation**: Mistakes in early subtasks cascade down the chain, potentially compounding errors 
- **Context Management**: Passing large intermediate outputs may exhaust the model’s context window, limiting chain 
- **Increased Latency**: Multiple prompts incur additional API calls, slowing response times


## 4.6 Tree of thoutghs


![image](https://github.com/user-attachments/assets/d86f73c9-7341-4dbf-955a-ed403ab5dfa3)



### References

1. “Self‑Consistency Improves Chain‑of‑Thought Reasoning in Language Models,” Xuezhi Wang et al., *NeurIPS* 2022.   
2. Prompt Engineering Guide: Self‑Consistency.   





```
Pranab Sahoo et al., A Systematic Survey of Prompt Engineering in Large Language Models, 2024 
ARXIV
.

Jason Wei et al., “Chain‑of‑Thought Prompting Elicits Reasoning in Large Language Models,” NeurIPS 2022 
ARXIV
.

Sander Schulhoff et al., The Prompt Report: A Systematic Survey of Prompting Techniques, 2024 
ARXIV
.

Amir Aryani, “8 Types of Prompt Engineering,” Medium, Dec 2023 
MEDIUM
.

“ReAct Pattern,” Wikipedia, 2024 
WIKIPEDIA
.

Xuezhi Wang et al., “Self‑Consistency Improves Chain‑of‑Thought Reasoning in Language Models,” 2022 
ARXIV
.

Shunyu Yao et al., “Tree of Thoughts: Deliberate Problem Solving with Large Language Models,” NeurIPS 2023 
ARXIV
.

“Retrieval‑Augmented Generation,” Wikipedia, 2024 
WIKIPEDIA
.

Yongchao Zhou et al., “Large Language Models Are Human‑Level Prompt Engineers (APE),” 2022 
ARXIV

Prompt Engineering Guide: Zero-Shot Prompting

Wei et al., 2022. "Finetuned Language Models Are Zero-Shot Learners."
Min et al., 2O22. "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?"
Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer

Xu and al., 2024. "Evaluating Large Language Models on Spatial Tasks: A Multi-Task Benchmarking Study"
Liuchang Xu, Shuo Zhao, Qingming Lin, Luyao Chen, Qianqian Luo, Sensen Wu, Xinyue Ye, Hailin Feng, Zhenhong Du

Wang et al., 2022“Self‑Consistency Improves Chain‑of‑Thought Reasoning in Language Models,” Xuezhi Wang et al., *NeurIPS* 2022.   

```

