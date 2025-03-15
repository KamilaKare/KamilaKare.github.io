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
 


#### **Decoder-Only Models**  
Decoder-only architectures utilize only the Transformer’s decoder stack. They are designed for autoregressive text generation, where each token is generated sequentially.  
- **Example:** **GPT (Generative Pre-trained Transformer)**  
- **Use Cases:**  
  - Causal language modeling  
  - Text generation (e.g., chatbots, story writing)  
  - Code generation  

#### **Encoder-Decoder Models (Seq2Seq Transformers)**  
These models leverage both the encoder and decoder stacks, making them particularly effective for sequence-to-sequence (seq2seq) tasks where an input sequence needs to be transformed into an output sequence.  
- **Example:** **T5 (Text-to-Text Transfer Transformer)**  
- **Use Cases:**  
  - Machine translation  
  - Text summarization  
  - Question answering  

Each of these variations leverages the Transformer’s self-attention mechanism while modifying its structure to optimize performance for different natural language processing (NLP) tasks.  


Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 4171–4186).
arXiv:1810.04805

