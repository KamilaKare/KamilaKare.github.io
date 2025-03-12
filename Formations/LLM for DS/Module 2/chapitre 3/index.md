y rajouter directement bert, llama et t5?
### 3.3. Variations of the Transformer  

While the original Transformer model features both an encoder and a decoder, different adaptations of the architecture have been developed for specific tasks. These variations fall into three main categories:  

#### **Encoder-Only Models**  
Encoder-only architectures focus solely on the Transformer’s encoder stack, making them ideal for classification and token-level tasks.  
- **Example:** **BERT (Bidirectional Encoder Representations from Transformers)**  
- **Use Cases:**  
  - Masked language modeling (MLM)  
  - Text classification  
  - Named entity recognition  
  - Sentence embedding  

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
