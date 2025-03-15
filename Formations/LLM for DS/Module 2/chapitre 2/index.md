---
layout: default
title: Transformer Architecture
---

# Chapter 2: Transformer Architecture

With the key **building blocks** (multi-head attention, feed-forward networks, positional encoding, and residual connections + layer normalization) now in hand, we can explore how they come together to form the **Transformer**. Introduced by *Vaswani et al. (2017)*, the Transformer has become a cornerstone of modern deep learning for **sequence-to-sequence** tasks in NLP, speech, and beyond.

---

## 2.1. Model Overview

{% include image.html src="https://github.com/user-attachments/assets/00c6e273-9175-4083-9b3a-97089aebeb40" alt="The Transformer Architecture" caption="The Transformer Architecture." %}

A **Transformer** processes sequences in parallel, using **attention** mechanisms instead of recurrence or convolution to capture dependencies. It typically consists of:

1. **Encoder**: A stack of identical layers that encodes the source sequence into a set of continuous representations.  
2. **Decoder**: Another stack of identical layers that decodes those representations into an output sequence, often one token at a time.

Each **encoder** and **decoder** layer includes:
- **Multi-Head Attention** (self-attention in the encoder, self-attention + Masked attention + cross-attention in the decoder),
- **Feed-Forward** sub-layer,
- **Residual Connections** and **Layer Normalization** around each sub-layer.

Additionally, **positional encoding** is added to each token’s embedding to preserve sequence order.




---

## 2.2. The Encoder

### 2.2.1 Encoder Architecture

An **encoder** is composed of \(N\) identical layers, where each layer contains two main sub-layers:

1. **Self-Attention Sub-Layer**  
   - The tokens in the **same sequence** attend to each other, determining the context for each position.  
   - Residual connection and layer normalization wrap around this sub-layer.

2. **Feed-Forward Sub-Layer**  
   - A position-wise feed-forward network (often a 2-layer MLP with ReLU or GELU).  
   - Another residual connection and layer normalization follow this sub-layer.

## 2.3 The Decoder Architecture
A decoder also consists of $𝑁$ identical layers, but each layer has three sub-layers:

### 2.3.1 Masked Self-Attention
The decoder attends to its own partial output sequence, but uses a mask to avoid looking at future tokens.
Followed by a residual connection and layer normalization.

{% include image.html src="https://github.com/user-attachments/assets/66d2d9eb-42df-4cbb-bbdb-e16462296feb" alt="The Transformer Architecture" caption="The Transformer Architecture." %}

In this masked self-attention mechanism:

- Token 1 (query) can attend to key 1 only.
- Token 2 (query) can attend to key 1 and key 2, but not key 3 or key 4.
- Token 3 (query) can attend to key 1, key 2, and key 3, but not key 4.
- Token 4 (query) can attend to all previous keys (1, 2, 3, and 4).
  
This follows a causal structure, where each token cannot attend to future tokens, ensuring that predictions are made sequentially without "seeing" future information.



### 2.3.2 Cross-Attention (Encoder-Decoder Attention)

In cross-attention, the decoder attends to the encoder’s output representations, allowing it to incorporate contextual information from the input sequence. This mechanism helps the decoder generate relevant outputs based on the encoded input.

A residual connection is applied around the attention mechanism, followed by layer normalization, ensuring stable gradients and improved training dynamics.

### 2.3.3 Feed-Forward Sub-Layer

Same position-wise feed-forward network as in the encoder. Again, residual + layer normalization.

### 2.3.4 End-to-End Flow

#### **Encoder:**
- **Input tokens**: $ x_1, \dots, x_m $
- **Embedding + positional encoding**  
  → Pass through $N$  encoder layers  
  → Produces **encoder outputs** $h$.

#### **Decoder:**
- **Partial target tokens**: $y_{<t}$
- **Embedding + positional encoding**  
  → Pass through $N$ decoder layers, using both:  
  - **Masked self-attention** on  $y_{<t}$ 
  - **Cross-attention** to $h$  
  → Outputs a **hidden representation** $z_t$.

#### **Projection + Softmax:**
- $z_t$ is projected to **logits** for each vocabulary token.
- **Softmax** produces a **probability distribution** over the next possible token.

### 2.3. Variations of the Transformer  

While the original Transformer model features both an encoder and a decoder, different adaptations of the architecture have been developed for specific tasks. These variations fall into three main categories:  

### 2.3.1 **Encoder-Only Models**  
Encoder-only architectures focus solely on the Transformer’s encoder stack, making them ideal for classification and token-level tasks.  
- **Examples:** **BERT** , **ModernBERT**, **RoBERTa**, **DistilBERT** etc 
- **Use Cases:**  
  - Masked language modeling (MLM)  
  - Text classification  
  - Named entity recognition  
  - Sentence embedding

#### 2.3.1.1 **BERT (Bidirectional Encoder Representations from Transformers)**

{% include image.html src="https://github.com/user-attachments/assets/f696d948-897c-4caa-9eff-f12d3d87faca" alt="The BERT Architecture" caption="The BERT Architecture." %}

#### Overview of BERT
**BERT** was introduced by researchers at Google in 2018. The name stands for *Bidirectional Encoder Representations from Transformers*, highlighting its core idea: using **bidirectional context** from the **encoder** portion of the Transformer architecture. Unlike earlier models (e.g., GPT) that read text primarily from left-to-right, BERT processes the entire sentence at once, capturing context from both directions.

#### Training Objectives
BERT’s success is largely due to its two main pre-training tasks:

1. **Masked Language Modeling (MLM)**
   - **What it does:** Randomly masks (i.e., replaces) some percentage (usually 15%) of the tokens in the input with a special `[MASK]` token. BERT then attempts to predict the original tokens.
   - **Why it matters:** Forces BERT to learn **deep bidirectional context**, as the model must use information from both left and right sides of a masked token to infer the missing word.

2. **Next Sentence Prediction (NSP)**
   - **What it does:** Takes pairs of sentences and trains the model to predict whether the second sentence logically follows the first (*IsNext*) or is just a random sentence (*NotNext*).
   - **Why it matters:** Encourages the model to learn **sentence-level relationships**, which is crucial for tasks like question-answering or text classification that require understanding how sentences connect.

> **Note:** Later research (e.g., RoBERTa) showed that NSP is not always essential for downstream performance. However, BERT’s original design included NSP, and it still serves as a relevant example of how to incorporate sentence-level tasks in pre-training.

#### Architecture Highlights
1. **Encoder-Only Stack**  
   - BERT consists of multiple **Transformer encoder** layers (commonly 12 or 24), each with **multi-head self-attention** and **feed-forward** sub-layers.
   - There is no separate decoder stack, making BERT well-suited for tasks that do not require auto-regressive generation (e.g., classification, token labeling).

2. **Bidirectional Attention**  
   - Each encoder layer attends to all tokens in the input sequence at once, allowing BERT to gather context from **both left and right** simultaneously.

3. **Positional Embeddings**  
   - Like other Transformers, BERT uses **positional embeddings** to represent token positions in the input sequence. This helps the model understand the order of tokens.

4. **Special Tokens**  
   - **`[CLS]`**: Added at the start of every sequence; its final hidden state is often used for **classification tasks**.  Indeed, the final hidden state corresponding to the [CLS] token is designed to capture an aggregated representation of the entire input sequence. This means that after processing the input through BERT's layers, the vector associated with [CLS] embodies the contextual information of the whole sequence.
   - **`[SEP]`**: Used to separate multiple sentences or segments within the input.

#### Fine-Tuning BERT
After BERT is pre-trained on large text corpora (e.g., Wikipedia + BookCorpus), it can be **fine-tuned** on a variety of downstream tasks by simply adding a small classification head or other task-specific layers on top. Fine-tuning typically takes far fewer training steps than the initial pre-training, thanks to BERT’s strong contextual knowledge.

#### Common Downstream Tasks
1. **Text Classification** (e.g., sentiment analysis)  
   - Add a **fully connected layer** on top of the `[CLS]` token output and train for classification.
2. **Named Entity Recognition (NER)**  
   - Use the **token-level outputs** for each position and classify whether each token is a named entity.
3. **Question Answering**  
   - Predict the start and end token indices of the answer within a passage.
4. **Sentence Pair Tasks** (e.g., natural language inference, sentence similarity)  
   - Use the **`[CLS]`** representation, or a combination of both sentence embeddings, to determine relationships between sentences.

BERT has significantly **influenced modern NLP** by demonstrating the effectiveness of pre-training large Transformer-based models on massive corpora, then fine-tuning on specific tasks. It paved the way for many variants (e.g., RoBERTa, DistilBERT, ALBERT) that optimize or extend BERT’s core ideas.

#### Practical Considerations
1. **Computational Resources**  
   - BERT-base has ~110M parameters, while BERT-large has ~340M. Fine-tuning them can be resource-intensive, requiring GPUs or TPUs.
2. **Hyperparameter Tuning**  
   - Learning rates, batch sizes, and training epochs can drastically affect performance.
3. **Handling Long Sequences**  
   - Standard BERT models have a **512 token** limit. Techniques like **Longformer** or **BigBird** address longer context windows.
4. **Model Distillation**  
   - To deploy on resource-constrained environments (e.g., mobile devices), **DistilBERT** and other compression approaches reduce model size and inference latency.
 


### 2.3.2 **Decoder-Only Models**  
Decoder-only architectures utilize only the Transformer’s decoder stack. They are designed for autoregressive text generation, where each token is generated sequentially.  
- **Example:** **GPT (Generative Pre-trained Transformer)** , **Llama series**, **Mistral**, **Claude** etc
- **Use Cases:**  
  - Causal language modeling  
  - Text generation (e.g., chatbots, story writing)  
  - Code generation


#### **LLaMA 2 Architecture Overview**

{% include image.html src="https://github.com/user-attachments/assets/55fd2977-ced3-4a56-8b81-75b57eb2f335" alt="The Llama Architecture" caption="The Llama Architecture." %}

The figure above depicts a **single decoder block** in the LLaMA 2 model. This block is repeated **N** times (denoted as **Nx**) to form the full **decoder-only** Transformer architecture. Below is a breakdown of the key components and how they connect:

#### **1. Input Embeddings + Rotary Positional Encodings**
- **Token Embeddings**: Each input token is converted into a learnable embedding vector.  
- **Rotary Position Encodings**: Instead of fixed or absolute positional embeddings, LLaMA 2 applies **rotary** encodings to each token’s embedding, helping the model preserve relative ordering in a continuous way.

#### **2. RMS Norm**
- After combining embeddings and position encodings, LLaMA 2 applies an **RMS (Root Mean Square) Norm**.  
- **Why RMS Norm?** It provides a stable normalization alternative to LayerNorm, scaling hidden states based on their root mean square rather than mean and variance.

#### **3. Self-Attention (Grouped Multi-Query Attention) + KV Cache**
- **Grouped Multi-Query Attention**: A variation of multi-head attention where keys and values are shared (or grouped) among heads, reducing memory usage.  
- **Causal Masking**: Ensures each token only attends to itself and preceding tokens for autoregressive text generation.  
- **KV Cache**: Speeds up inference by caching Key-Value pairs from previous time steps, so the model doesn’t recompute them for each new token.
  
#### **4. RMS Norm + Feed-Forward (SwiGLU)**
- **RMS Norm**: Normalizes again before the feed-forward sub-layer.  
- **Feed-Forward (SwiGLU)**: A two-layer MLP using the **SwiGLU** activation function. This often improves training stability and model performance compared to ReLU or GELU.

#### **5. Residual Connections**
- Each sub-layer (Self-Attention, Feed-Forward) is wrapped with **residual connections**, allowing the model to pass information around these transformations and helping gradients flow more effectively.

#### **6. Output Layer**
- After **N** repeated decoder blocks, the final hidden states pass through a **Linear** projection to the vocabulary dimension, followed by a **Softmax** to produce probability distributions over possible next tokens.

This **decoder-only** design, combined with **causal masking** and **KV caching**, makes LLaMA 2 well-suited for **autoregressive** tasks such as text generation, dialogue systems, and more. Its use of **RMS Norm**, **rotary position encodings**, and **SwiGLU** are notable departures from classic Transformer implementations, contributing to LLaMA 2’s efficiency and performance. Moreover, this approach aligns with other large language models that rely on a decoder-only architecture, which uses unidirectional attention to generate text token by token. 

A dedicated chapter on fine-tuning these decoder-only models will explore how to adapt them for specific tasks, covering techniques such as LoRA (Low-Rank Adaptation), full fine-tuning, instruction tuning, and Reinforcement Learning with Human Feedback (RLHF) to enhance their performance on domain-specific applications.


### 2.3.3 **Encoder-Decoder Models (Seq2Seq Transformers)**  
These models leverage both the encoder and decoder stacks, making them particularly effective for sequence-to-sequence (seq2seq) tasks where an input sequence needs to be transformed into an output sequence.  
- **Example:** **T5 (Text-to-Text Transfer Transformer)**  
- **Use Cases:**  
  - Machine translation  
  - Text summarization  
  - Question answering

#### 2.3.3.1 **T5 (Text-to-Text Transfer Transformer)**

**T5** is a Transformer-based model from Google that adopts a full **encoder-decoder** architecture. It stands out by framing **all** natural language tasks into a unified *text-to-text* format, meaning both inputs and outputs are always strings.

#### Overview
- **Released by:** Google Research (Raffel et al., 2019)  
- **Core Idea:** Convert various NLP tasks (e.g., translation, summarization, classification) into a *text-to-text* paradigm.  
- **Why It Matters:** Simplifies the process of fine-tuning on multiple tasks—everything is handled by providing text input and expecting text output.

#### Architecture Highlights
1. **Encoder-Decoder Stack**  
   - **Encoder**: Processes the input sequence using multi-head self-attention.  
   - **Decoder**: Generates the output sequence in a **left-to-right** fashion, attending to both the encoder output and previously generated tokens.

2. **Multi-Head Attention**  
   - **Encoder Self-Attention**: Learns bidirectional context for input tokens.  
   - **Decoder Self-Attention**: Uses **causal masking** to predict the next token.  
   - **Cross-Attention**: Decoder attends to the encoder’s output, integrating contextual information from the input.

3. **Task Prefixes**  
   - Each task (e.g., “translate English to German”) is specified by a prefix, guiding T5 to produce the desired output format.

4. **Shared Vocabulary & Token Embeddings**  
   - T5 uses a **SentencePiece** tokenizer with a fixed vocabulary.  
   - The encoder and decoder share embeddings to reduce the total parameter count.
     
#### Pre-Training Objective: “Span Corruption”
- **Span Masking**: Instead of masking individual tokens, T5 **masks contiguous spans** of tokens in the input.  
- **Task:** The model predicts the missing tokens from these masked spans, effectively learning bidirectional context in the encoder while training the decoder to reconstruct the masked segments.

#### Fine-Tuning & Use Cases
- **Text Summarization**: Provide a prefix like `"summarize: "` followed by the document. The model outputs a concise summary.  
- **Machine Translation**: Use a prefix like `"translate English to French: "` followed by the English text.  
- **Question Answering**: Present a question and context text, instruct T5 with a prefix (e.g., `"question: … context: …"`).  
- **Classification**: Convert classification tasks into a text-to-text format (e.g., `"sst2 sentence: I loved this movie. sentiment: "`).

Because T5 treats all tasks in a unified manner, **fine-tuning** typically just involves changing the **task prefix** and training on a supervised dataset with input-output text pairs.

#### Model Variants
T5 comes in multiple sizes (e.g., **T5-Small**, **T5-Base**, **T5-Large**, **T5-3B**, **T5-11B**).  
- Larger variants capture **more complex patterns** but require **greater compute**.  
- Smaller variants are easier to **train** and **deploy** but may have lower performance.

#### Practical Considerations
1. **Compute Requirements**  
   - Large T5 models can be resource-intensive; smaller versions are often sufficient for many tasks.
2. **Prompt Engineering**  
   - Choosing the right prefix or instruction format can significantly affect performance.
3. **Continued Pre-training**  
   - Some tasks benefit from domain-specific continued pre-training (e.g., medical or legal corpora).
4. **Distillation & Pruning**  
   - Techniques like model distillation can reduce T5’s size for faster inference on limited hardware.

  

Each of these variations leverages the Transformer’s self-attention mechanism while modifying its structure to optimize performance for different natural language processing (NLP) tasks.  



Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 4171–4186).
arXiv:1810.04805

https://www.linkedin.com/posts/kamila-kare-phd-572a87112_genai-llm-activity-7219240280481312769-B1em/?originalSubdomain=fr

