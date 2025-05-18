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


### 2.1  Deep-dive on Critical Stages

#### 2.1.1  Ingest & Index  (offline / batch)

| Sub-step | What you actually do | Design choices & pitfalls |
|----------|---------------------|---------------------------|
| **1. Connect** | Read raw data from web crawlers, SharePoint, Confluence, PDFs, SQL dumps, S3, etc. | Avoid “one-shot” ETL scripts—use connectors that can run incrementally so you don’t re-process the whole corpus nightly. |
| **2. Clean & Normalize** | Strip boilerplate, remove HTML nav bars, fix encoding, merge tiny paragraphs. | Garbage-in → garbage-retrieved. Cleaning is *the* place to invest time. |
| **3. Chunk / Split** | Break documents into ~200-400-token windows with 10-20 token overlap. | <200 tokens: lost context; >500 tokens: harder to match. Tune per domain. |
| **4. Enrich Metadata** | Attach `title`, `url`, `date`, `author`, tags, security labels. | Metadata lets you filter by customer, language, region, ACL during retrieval. |
| **5. Embed** | Transform chunks → d-dim vectors (`text-embedding-3-large` or open-source BGE). | Use a *single* embedding model for both docs and queries. Re-embed when you upgrade models. |
| **6. Store** | Write vectors + metadata into FAISS, Milvus, Pinecone, Qdrant, or Elastic hybrid index. | • Pick **HNSW** or **IVF-Flat** for low-latency ANN.  <br>• Version your index dumps just like model weights. |
| **7. Refresh** | Schedule nightly or streaming jobs; mark deleted docs as tombstones; invalidate L2 caches. | Missing tombstones causes “ghost citations”—users click a link that 404s. |

> **Rule of thumb:** ~60 % of a production RAG team’s engineering time disappears in steps 2-4—everything after that is comparatively easy.

---

#### 2.1.3  Retrieval  (online / real-time)

1. **Query Pre-Processing**  
   * Spell-correction, language detection, PII scrubbing.  
   * Optionally call an LLM (“Query Rewriter”) to expand acronyms or clarify pronouns.

2. **Primary Retrieval**  
   * **Dense ANN search** over embeddings **OR** BM25 keyword search.  
   * Typical settings: `k_dense = 4-8`, `k_sparse = 10-20`.

3. **Hybrid Fusion (optional but recommended)**  
   * Normalize both score lists → merge with **Reciprocal Rank Fusion**  
     \(\text{RRF}(d) = \sum_{i}{\frac{1}{k + \text{rank}_i(d)}}\) with \(k≈60\).  
   * Boosts recall ~8-15 % with <10 ms extra latency on a 100 k doc corpus.

4. **Reranking**  
   * Cross-encoder such as `bge-reranker-large` re-scores top-N (N≤100).  
   * Costs one LLM call per candidate pair; cache results aggressively.

5. **Filtering & Diversity**  
   * Drop chunks that share the same URL/section to avoid redundancy.  
   * Apply metadata ACL filters (user tenancy, security level, language).

6. **Return Top-K Context**  
   * Final list usually **3-6 chunks** totalling ≤ ⅔ of the LLM’s context window.  
   * Pass along `source_id`, `vector_score`, and any `highlights` for UI.

> **Latency budget:** Aim for ≤ 100 ms P95 end-to-end retrieval (including rerank) to keep total chatbot latency under 1 s.

---

#### 2.1.6  Generation  (LLM & post-processing)

| Component | Detail | Best practice |
|-----------|--------|---------------|
| **Prompt Builder** | Format:  <br>`SYSTEM: you are …`<br>`CONTEXT: <doc id=1>…</doc> …`<br>`USER: {{question}}` | Use explicit XML/Markdown delimiters so the model can’t cite outside the block. |
| **Token Budgeting** | Reserve ≈ ⅓ of context for the answer itself. Trim context by score order if near the limit. | Truncating arbitrarily (“top-k=10 no matter what”) hurts faithfulness—prefer dynamic pruning. |
| **LLM Call** | GPT-4o, Claude-3 Opus, Mistral-Large, etc. Choose temperature 0-0.2 for factual QA. | Keep *one* LLM family in prod; multi-vendor adds evaluation overhead. |
| **Inline Reasoning** | Ask the model to think step-by-step in a hidden scratchpad, then produce a concise answer. | Use **`<scratchpad>`** tags and instruct the model to hide that part from the user—reduces hallucinations. |
| **Citations Formatting** | Numbered refs `[1]`, JSON array, or HTML anchors—pick what your front-end can render. | Teach the model with explicit examples *inside* the prompt. |
| **Answer Post-Processing** | • Regex to turn `[1]` into clickable links. <br>• PII redaction. <br>• Guardrails scan for disallowed content. | If any guard fails, either re-ask with `temperature=0` or display a refusal. |
| **Telemetry Emit** | Log `trace_id`, `prompt_hash`, `retrieved_ids`, latency, token counts, user rating. | Hash prompts so you can dedupe and visualize hot queries. |

Common failure modes → mitigations  
* **Citation drift** – model cites the wrong doc. → Shorter chunks + reranker.  
* **Answer too short** – model truncates mid-sentence. → Increase `max_tokens` and reduce context length.  
* **Toxic or private info appears** – missing guardrail. → Layer Giskard “LLM safety scan” after generation.

---

> **Remember:** RAG quality is the *minimum* of retrieval quality **and** generation faithfulness.  
> Over-investing in only one half leaves you with “retrieved garbage” or “grounded hallucinations.”

