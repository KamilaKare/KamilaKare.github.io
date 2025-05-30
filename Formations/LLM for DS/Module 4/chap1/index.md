---
layout: default
title: Retrieval Augmented Generation and Agents
---

# Chapter 1 · Retrieval-Augmented Generation (RAG)

> **Goal of the chapter**  
> Equip students with a deep, end-to-end understanding of why RAG matters, how a modern RAG stack is built, the main retrieval algorithms in play, how to measure quality with *Ragas* and *Giskard*, which advanced variants to watch, and what it really takes to run RAG in production.

---

## 1  Defintion

### Motivation
Large language models excel at pattern-matching over everything they saw during training—but that frozen knowledge stops the moment training ends. Whenever you question them about niche, fast-moving, or proprietary topics, they risk hallucinating, going out-of-date, or leaking private data if you try to fine-tune them. In short, LLMs need a reliable way to **look things up on-the-fly** instead of guessing.


![image](https://github.com/user-attachments/assets/2c67f2f5-dd71-4bd4-b2ae-cc20ab0861a0)

### Definition
Retrieval-Augmented Generation (RAG) is a hybrid architecture that pairs a fast **document retriever** with a **large language generator**.
At inference time, the retriever pulls the most relevant snippets from a curated knowledge store, and the generator weaves those snippets into a coherent, cited response—so the model can “look things up” instead of relying solely on its frozen parameters.

{% include image.html src="https://github.com/user-attachments/assets/2c67f2f5-dd71-4bd4-b2ae-cc20ab0861a0" alt="RAG " caption="RAG Architecture from Datacamp." %}

## Why RAG?

| Pain-point of vanilla LLMs | How RAG mitigates it |
| --- | --- |
| **Hallucinations** – the LLM fabricates facts | Retrieved passages ground answers in verifiable text |
| **Stale knowledge** – parameters freeze at train-time | Retrieval hits a *live* index that you can refresh continuously |
| **Data-privacy trade-off** – fine-tuning on private corpora leaks IP | Keep weights untouched; keep data in your own vector DB |
| **Token-window limits** – long documents get truncated | Retriever selects only the most relevant snippets |

RAG has therefore been adopted by every hyperscaler (Microsoft, Google, Amazon, Nvidia) and is now considered *table stakes* for factual enterprise chatbots and AI search experiences. 

---

## 2  High-Level Architecture

{% include image.html src="https://github.com/user-attachments/assets/7418ebcb-abdc-4f0e-add7-69e6538dfd85" alt="RAG " caption="RAG Architecture " %}

### Ingest & Index

The life of every document begins **offline**. Raw user assets—PDFs, HTML pages, slide decks, log files—are first collected by connectors or crawlers. They are scrubbed for boilerplate, unified to UTF-8, and normalized so that headings, code blocks, and tables survive intact. Each cleaned document is then **chunked** into overlapping windows of roughly 200–400 tokens; this window is small enough for precise retrieval yet large enough to preserve local context. Every chunk is passed through the *same* embedding model you will later use for queries, yielding a fixed-length numeric vector that captures its semantics. The vectors are persisted alongside metadata (title, URL, timestamp, ACL labels) and bulk-loaded into the Vector DB, which builds an approximate-nearest-neighbour (ANN) index—HNSW or IVF-Flat are common—so that similarity search later runs in tens of milliseconds. This entire pipeline is versioned and usually scheduled to run incrementally: new or changed source files are embedded and merged while deleted files are tomb-stoned, keeping the index perfectly aligned with the live knowledge base.

### Retrieval

The retrieval phase is online and latency-critical. When a user submits a question, the query text is immediately embedded by the exact same model used during indexing, guaranteeing that query and document vectors inhabit the same semantic space. The resulting query vector is fired at the Vector DB, which performs a k-nearest-neighbour search to surface the most similar chunks— i.e. the **relevant context**. Typical settings are k ≈ 4–8 for dense search, optionally fused with a sparse BM25 pass to capture exact keywords. Returned results honour any metadata filters (language, customer tenancy, security tier) applied at query time, ensuring the system never leaks information a user should not see. Because the index has already pruned the universe down to these few passages, you preserve both latency and token budget for what comes next.

### Generation

The retrieved passages flow into the blue “Augmentation” box, where they are concatenated with the original user question to form the **augmented prompt**. Delimiter tags or Markdown fences isolate each passage so the LLM can cite them unambiguously. A prompt-builder also injects system instructions—tone, audience, citation format—and trims the context if it threatens to exceed two-thirds of the model’s token window. This package is handed to the green neural-brain icon, your LLM. With temperature set low (0–0.2 for factual QA), the model reasons over the supplied evidence and crafts a response that quotes or paraphrases—but does not hallucinate beyond—the retrieved text. The answer, now grounded and optionally decorated with `[1]`-style citations, is streamed back to the user interface. Any final post-processing—linking citation numbers to source URLs, masking residual PII, or applying toxicity guards—happens here before the **Response** box completes the round-trip from raw documents to trustworthy, on-demand answers.

Common failure modes → mitigations  
* **Citation drift** – model cites the wrong doc. → Shorter chunks + reranker.  
* **Answer too short** – model truncates mid-sentence. → Increase `max_tokens` and reduce context length.  
* **Toxic or private info appears** – missing guardrail. → Layer Giskard “LLM safety scan” after generation.

---

> **Remember:** RAG quality is the *minimum* of retrieval quality **and** generation faithfulness.  
> Over-investing in only one half leaves you with “retrieved garbage” or “grounded hallucinations.”
>
> 
## 3 Retrieval Strategies

| Strategy                         | How it works                           | Strengths                          | Limitations                           | Good defaults                     |                      |
| -------------------------------- | -------------------------------------- | ---------------------------------- | ------------------------------------- | --------------------------------- | 
| **Dense** semantic search        | Embed query + docs → cosine similarity | Captures synonyms & paraphrase     | Can miss exact phrases; large index   | `text-embedding-3-large`, k = 4–8 |    
| **Sparse** lexical search (BM25) | Keyword TF-IDF scoring                 | Precise keywords; fast             | Ignores semantics; brittle to wording | Elastic BM25, top\_k ≈ 10         |           
| **Hybrid** (dense ⊕ sparse)      | Combine scores or union lists          | Best of both worlds, higher recall | Extra latency & tuning α              | Reciprocal Rank Fusion α≈0.4      |                   
| **Rerankers**                    | Cross-encoder scores top-N             | Great precision                    | Expensive (quadratic in N)            | `bge-reranker-large`, N ≤ 100     |                  
| **Knowledge-graph + dense**      | Retrieve KG triplets then text         | Handles entities, joins            | Need KG construction                  | HyPA-RAG pipeline             
| **Self-query / Iterative**       | LLM reformulates query iteratively     | Better long/complex Qs             | Higher cost                           | *Self-RAG*, ReACT loop            |                    

source
[1]: https://aclanthology.org/2025.naacl-industry.79.pdf?utm_source=chatgpt.com "[PDF] HyPA-RAG: A Hybrid Parameter Adaptive Retrieval-Augmented ..."

## 4 · Evaluating a Retrieval-Augmented Generation (RAG) System  

A RAG pipeline has **two tightly-coupled engines**—retrieval and generation—so a complete evaluation framework must score *both*, plus the glue between them. Below are the core metrics most teams track, grouped by layer. Tools such as **Ragas** (quality/groundedness) and **Giskard** (robustness/safety) compute many of these out-of-the-box, but the definitions are model-agnostic.

---

### 4.1  Retrieval Metrics  

| Metric | What it measures | Definition |
|--------|------------------|------------|
| **Recall @ *k*** | Coverage | Fraction of questions for which a relevant passage appears within the first *k* retrieved chunks. A drop here means the LLM never even *sees* the right evidence. |
| **MRR (Mean Reciprocal Rank)** | Early relevance | For each question, take **1 / rank** of the *first* relevant chunk; average across the dataset. Penalises answers buried deep in the list. |
| **nDCG (Normalised Discounted Cumulative Gain)** | Rank quality with graded relevance | Weighs highly relevant passages more and discounts lower ranks logarithmically; scale 0–1. Useful when passages have graded relevance labels, not just binary. |
| **Precision @ *k*** | Noise in context | Ratio of the top-*k* passages that are actually relevant. High precision keeps token budgets lean and reduces hallucination risk. |

---

### 4.2  Generation Metrics  

| Metric | Aspect | Definition |
|--------|--------|------------|
| **Fluency** | Readability | Human or automated rating (1–5) of grammar, style, and clarity. |
| **ROUGE-L / BLEU** | Content overlap | Longest Common Subsequence (ROUGE-L) or n-gram overlap (BLEU) between generated answer and a gold reference. Mostly useful in closed-domain FAQs where references exist. |
| **Answer Relevancy (Semantic Similarity)** | Topicality | Cosine similarity between answer and gold reference in embedding space. Captures paraphrases better than n-gram metrics. |

---

### 4.3  Groundedness / Faithfulness Metrics  

| Metric | Question it answers | Definition |
|--------|---------------------|------------|
| **Faithfulness** | “Does every claim trace back to the provided context?” | Proportion of answer statements that are fully supported by at least one retrieved passage. Ragas computes this via an LLM-as-judge. |
| **Context Precision** | “How much of the supplied context was actually used?” | Tokens from retrieved chunks that appear (directly or paraphrased) in the answer divided by total context tokens. Low values signal wasted prompt space. |
| **Context Recall** | “Did the model *forget* important evidence?” | Portion of relevant information in the retrieved passages that the answer successfully incorporates. |
| **Retrieval Recall** | Bridging metric | Checks whether the retriever surfaced any passage that truly contains the answer. If this is zero, all downstream metrics collapse. |

---

### 4.4  Trust-&-Safety / Robustness Metrics  

| Metric | Risk addressed | Definition |
|--------|----------------|------------|
| **Prompt-Injection Vulnerability** | Jail-breaks, policy bypass | Ratio of crafted adversarial prompts that succeed in altering system instructions. Giskard ships pre-built attack libraries and pass/fail scoring. |
| **PII Leak Score** | Privacy | Percentage of answers that expose personally identifiable information when they should not. |
| **Toxicity / Harassment** | Brand safety | Probability that generated text exceeds a toxicity threshold (Perspective API or an open-source classifier). |
| **Out-of-Domain Robustness** | Stability | Degradation in core metrics (recall, faithfulness) when queries are noisy, multilingual, or adversarial. |

---

### 4.5  How the Tools Fit In  

* **Ragas** focuses on the **quality triad**—retrieval accuracy, answer relevancy, and faithfulness—returning a single report card your CI pipeline can gate on (e.g., *faithfulness ≥ 0.80*).  
* **Giskard** layers **robustness and safety** on top, scanning your entire pipeline for prompt-injection holes, PII leaks, and toxic outputs; it also generates synthetic QA sets to stress-test rare edge cases.

> **Rule of thumb:** ship only when *both* retrieval recall and faithfulness stay high—otherwise you risk “grounded-but-wrong” or “hallucination-free-but-irrelevant” answers. Continuous monitoring of these metrics in production is as important as the initial benchmark.


## 5 · Production Considerations  

### Latency budgets  
Aim for **≤ 1 s end-to-end** for user-facing chat. A typical slice is  
* 50–100 ms* for ANN search, *50–150 ms* for reranking, the rest for the LLM call.  
Use in-memory HNSW or IVF-Flat indexes, warm LLM worker pools, batch reranker
inputs, and stream the first tokens as soon as decoding starts.

### Cost control  
Large embedding models often dominate Total Cost of Ownership.  
* Use smaller per-token models for static corpora and reserve the pricey,
high-dimensional embeddings for hot or mission-critical collections.  
* Shrink *k* with metadata pre-filters and exploit semantic caches.  
* Schedule off-peak re-indexing jobs.

### Freshness & data quality  
Run **incremental ingests**—only new or modified documents are (re)embedded,
while deletions are tomb-stoned. This prevents stale answers without rebuilding the
whole index.

### Monitoring & observability  
Log the tuple **(query, retrieved_ids, LLM_version, latency, user_rating)** for every
request. Dashboards that track Recall@k, faithfulness, and token usage catch
regressions before users notice.

### Robustness & safety  
Automate **prompt-injection, PII-leak, and toxicity scans** in CI with tools such as
Giskard. Block deployment on any critical finding and provide refusal fallback
paths when guardrails trigger.

### Versioning & CI/CD  
Version **code, embeddings, index snapshot, and evaluation dataset together**.  
Gate promotion on targets like “RAGAS faithfulness ≥ 0.80 **and**
retrieval recall ≥ 0.90” to ensure neither half regresses.

---
## 5 · Advanced RAG Variants  

RAG is no longer just “retrieve and stuff.” As use cases get more complex and expectations rise, production-grade systems evolve through three levels of sophistication: **Table Stakes**, **Advanced Retrieval**, and **Agentic Behavior**. Beyond that, emerging directions like **GraphRAG** and **Multimodal RAG** extend RAG across modalities and structured data.

---

###  Table Stakes  
These are the minimum required to make a RAG pipeline usable, relevant, and efficient.

- **Better Parsers**: Extract structured content (tables, lists, sections) accurately from PDFs, HTML, or wikis—poor structure leads to poor chunks.
- **Chunk Sizes**: Use ~200–400 tokens with overlap. Small chunks improve precision, larger ones help with context continuity—tune by domain.
- **Hybrid Search**: Combine dense semantic retrieval with sparse BM25 keyword match. Hybrid recall often beats either alone.
- **Metadata Filters**: Index metadata like language, source, section type, or customer scope to enable ACLs and filtered retrieval.

---

###  Advanced Retrieval  
These methods optimize what gets retrieved, how it's scored, and how it’s composed before prompting the LLM.

- **Reranking**: Use a cross-encoder to reorder top-k chunks from initial retrieval. Improves precision and citation faithfulness.
- **Recursive Retrieval**: Use intermediate queries (e.g. decomposed questions) to refine or expand the result set in multiple passes.
- **Embedded Tables**: Encode tabular data alongside text so LLMs can reason over facts, specs, or structured answers.
- **Small-to-Big Retrieval**: Start with granular chunks (sentences), then retrieve full sections or documents for broader grounding.
- **GraphRAG**: Query a **knowledge graph** (via Cypher or SPARQL), retrieve triples, and insert those into the prompt. Ideal for entity-rich or relational data (e.g., product graphs, biomedical KGs).

---

###  Agentic Behavior  
At this level, the system behaves more like a **planner** or **autonomous assistant**, deciding what to retrieve, how to route, and how to synthesize across multiple steps.

- **Routing**: Direct a query to the right retriever, database, or prompt template (e.g., product vs. support vs. internal doc).
- **Query Planning**: Break a complex or ambiguous query into subquestions, run retrieval for each, then stitch together the final answer.
- **Multi-doc Agents**: Agents that reason over multiple sources, verify consistency, and optionally revisit retrieval as they compose multi-step answers.

---

###  Multimodal RAG (Emerging)  
RAG pipelines aren’t limited to text anymore.

- **Multimodal RAG**: Retrieve and ground answers on **non-text content** like images, tables, audio transcripts, or even code.  
  Use models like **CLIP**, **BLIP-2**, or **SigLIP** to embed visual or tabular data, then prompt a multimodal LLM to generate responses.

Use case examples:
- “What’s different between these GPU diagrams?” → image+text
- “Summarize the key points from this video transcript and graph.” → audio+structured+text

---

### Choosing What to Implement  

| System Maturity | Recommended Focus |
|------------------|--------------------|
| MVP / Pilot | **Table Stakes**: chunking, hybrid search, metadata filters |
| Scaling Accuracy | **Advanced Retrieval**: rerankers, recursive strategies, GraphRAG |
| Complex Workflows | **Agentic Behavior**: routing, planning, multi-doc agents |
| Multimodal Content | **Multimodal RAG**: images, structured data, audio transcripts |

---

> As models gain larger context windows and stronger reasoning, the bottleneck shifts from “model limitations” to “retrieval strategy.”  
> Investing in advanced RAG makes your system **more accurate, safer, and future-ready**.
