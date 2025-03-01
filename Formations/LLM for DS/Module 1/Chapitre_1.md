
# What is Generative AI

## 1.1 Introduction & Definitions 

### Defining Artificial Intelligence (AI)

Before diving into generative AI, it's essential to understand what artificial intelligence is. AI refers to algorithms and systems designed to mimic aspects of human intelligence. These technologies employ various techniques—from statistical methods to deep neural networks—to simulate cognitive functions such as:

- **Perception:** Interpreting and analyzing data from the environment.
- **Learning:** Adapting and improving from experience.
- **Decision-Making:** Evaluating situations to choose optimal actions.
- **Problem Solving:** Finding solutions to complex challenges.

### Transitioning to Generative AI

Building on these capabilities, AI has evolved into more sophisticated applications. **Generative AI (Gen AI)**, in particular, focuses on creating new content—be it text, images, audio, or even code—using models trained on large datasets. This innovative field not only enhances creative processes but also drives advancements in areas like personalized content creation, design innovation, and automated problem-solving.

### Contrasting Generative AI with Discriminative AI  

To better understand Generative AI, it's useful to contrast it with **Discriminative AI**.  

- **Discriminative AI**  aims to differentiate between different classes in the data. It focuses on making classifications or predictions. Given input data $X$, it assigns labels (Y) or estimates probabilities ($\mathbb{P} (Y|X)$).  
  - Example: A spam filter classifies emails as **spam** or **not spam**.  
  - Common models: Logistic Regression, Decision Trees, Support Vector Machines (SVMs), and traditional Neural Networks.  

- **Generative AI**, on the other hand, creates new data rather than just classifying existing data. It aims to understand and capture the underlying distribution of the data ($\mathbb{P} (X, Y)$) in order to generate new data points that are similar to the original dataset.
  - Example: A generative model can produce entirely new emails that resemble human-written ones.  
  - Common models: GANs (Generative Adversarial Networks), VAEs (Variational Autoencoders), and Large Language Models (LLMs) like GPT and BERT.  

While **Discriminative AI** is excellent for structured problems (e.g., fraud detection, medical diagnosis), **Generative AI** unlocks new possibilities in creativity, automation, and data augmentation.  

---
### Why Now?  

Generative AI has existed in some form for decades, but it has recently seen explosive growth. This is due to three key factors:  

1. **Growth in Compute Power**  
   - The rise of specialized hardware (GPUs, TPUs) has made training deep learning models much faster and more efficient.  
   - Cloud computing allows for large-scale AI training without needing expensive infrastructure.  

2. **Large Datasets**  
   - The internet provides vast amounts of text, images, and videos, which are essential for training generative models.  
   - Advances in data curation and preprocessing enable models to learn from diverse, high-quality sources.  

3. **Advanced Architectures: The Rise of Transformers**  
   - In 2017, the paper *"Attention Is All You Need"* introduced the **Transformer architecture**, which revolutionized AI by enabling better handling of long-range dependencies in data.  
   - Transformers power modern AI models like **GPT (OpenAI), BERT (Google), Mistral, and Llama**, enabling superior language understanding and generation.  

These advancements have made Generative AI **scalable, efficient, and accessible**, leading to real-world applications across industries, from marketing to healthcare to software development.  

**Why It Matters for Business**  
- Personalized marketing, automated content creation, design prototypes, and more.  
- Unlocking new possibilities (e.g., generative design, product innovation).

#### Suggested Slide Content:
- **Slide 1:** Title: “What is Generative AI?”
- **Slide 2:** Bullet points: Definition, Key Capabilities, Examples
- **Slide 3:** Real-world impacts (brief mention of ChatGPT, Midjourney, Copilot, etc.)

---

## 1.2 Evolution of AI: From Rule-Based to Transformers (10 min)

### Early AI & Rule-Based Systems
- Expert systems, if-then rules, symbolic logic.
- Limited by hand-crafted knowledge.

### Classical Machine Learning
- Supervised vs. unsupervised.
- Feature engineering, smaller datasets.
- Examples: logistic regression, decision trees, SVMs.

### Deep Learning
- Emergence of neural networks, CNNs for images, RNNs for sequences.
- Rise of big data and GPU computing.

### Transformers & Large Language Models
- 2017: “Attention Is All You Need” paper → The Transformer.
- Multi-head attention, parallelization, ability to handle large contexts.
- Examples: GPT, BERT, T5, Mistral, Llama.

#### Key Talking Points:
- **Attention Mechanism:** Why it outperforms RNNs for long sequences.
- **Scaling:** More parameters → more emergent capabilities (few-shot learning).
- **Cultural Moment:** ChatGPT & others bring LLMs mainstream.

#### Suggested Slide Content:
- **Slide 4:** Timeline: Rule-based (80s) → ML (90s-00s) → Deep Learning (2010s) → Transformers (2017+)
- **Slide 5:** Key differences: Feature engineering vs. representation learning vs. attention-based learning

---

## 1.3 Key Differences: Classical ML vs. Deep Learning vs. LLMs (5 min)

### Classical ML
- Often smaller datasets, manual feature engineering.
- Good for structured data (tabular).

### Deep Learning
- Representation learning from large data.
- CNNs for vision, RNNs for sequences.

### LLMs
- Pretrained on massive text corpora.
- Capable of text generation, summarization, translation, etc.
- Zero-/few-shot performance (less labeled data needed).

#### Suggested Slide Content:
- **Slide 6:** Table comparing Classical ML, Deep Learning, LLMs (data size, interpretability, typical tasks).
- **Slide 7:** Summarize real-world usage: ML for fraud detection, deep learning for image recognition, LLM for chatbots & content generation.
