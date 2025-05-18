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
###
| Strategy                         | How it works                           | Strengths                          | Limitations                           | Good defaults                     |                      |
| -------------------------------- | -------------------------------------- | ---------------------------------- | ------------------------------------- | --------------------------------- | -------------------- |
| **Dense** semantic search        | Embed query + docs → cosine similarity | Captures synonyms & paraphrase     | Can miss exact phrases; large index   | `text-embedding-3-large`, k = 4–8 |                      |
| **Sparse** lexical search (BM25) | Keyword TF-IDF scoring                 | Precise keywords; fast             | Ignores semantics; brittle to wording | Elastic BM25, top\_k ≈ 10         |                      |
| **Hybrid** (dense ⊕ sparse)      | Combine scores or union lists          | Best of both worlds, higher recall | Extra latency & tuning α              | Reciprocal Rank Fusion α≈0.4      |                      |
| **Rerankers**                    | Cross-encoder scores top-N             | Great precision                    | Expensive (quadratic in N)            | `bge-reranker-large`, N ≤ 100     |                      |
| **Knowledge-graph + dense**      | Retrieve KG triplets then text         | Handles entities, joins            | Need KG construction                  | HyPA-RAG pipeline                 | ([ACL Anthology][1]) |
| **Self-query / Iterative**       | LLM reformulates query iteratively     | Better long/complex Qs             | Higher cost                           | *Self-RAG*, ReACT loop            |                      |

[1]: https://aclanthology.org/2025.naacl-industry.79.pdf?utm_source=chatgpt.com "[PDF] HyPA-RAG: A Hybrid Parameter Adaptive Retrieval-Augmented ..."


