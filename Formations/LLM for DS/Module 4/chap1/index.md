---
layout: default
title: Retrieval Augmented Generation and Agents
---

# Chapter 1 · Retrieval-Augmented Generation (RAG)

> **Goal of the chapter**  
> Equip students with a deep, end-to-end understanding of why RAG matters, how a modern RAG stack is built, the main retrieval algorithms in play, how to measure quality with *Ragas* and *Giskard*, which advanced variants to watch, and what it really takes to run RAG in production.

---

## 1  Why RAG?

| Pain-point of vanilla LLMs | How RAG mitigates it |
| --- | --- |
| **Hallucinations** – the model fabricates facts | Retrieved passages ground answers in verifiable text |
| **Stale knowledge** – parameters freeze at train-time | Retrieval hits a *live* index that you can refresh continuously |
| **Data-privacy trade-off** – fine-tuning on private corpora leaks IP | Keep weights untouched; keep data in your own vector DB |
| **Token-window limits** – long documents get truncated | Retriever selects only the most relevant snippets |

RAG has therefore been adopted by every hyperscaler (Microsoft, Google, Amazon, Nvidia) and is now considered *table stakes* for factual enterprise chatbots and AI search experiences. :contentReference[oaicite:0]{index=0}

---

## 2  High-Level Architecture

```text
              ┌───────────────────────────┐
   Offline    │ 1  INGEST & INDEX         │  ⮜  connectors, scrapers
  ----------► │   – load → clean → chunk  │
              │   – embed → store in DB   │
              └────────▲──────────────────┘
                       │                   nightly refresh
┌─────────────── user query ───────────────┐
│               ▼                          │           ─── online path
│ 2  QUERY UNDERSTANDING                   │  ↓ spell-fix, rewrite, classification
│ 3  RETRIEVER  (dense / sparse / hybrid)  │  ↓ top-k ids
│ 4  RERANK / FUSION (optional)            │  ↓ re-ordered ctx
│ 5  PROMPT BUILDER + GUARDRAILS           │  ↓ context window
│ 6  GENERATIVE MODEL (LLM)                │  ↓ draft answer
│ 7  POST-PROCESS & CITATIONS              │  ↓ final answer
│ 8  TELEMETRY & FEEDBACK LOOP             │  ↑ logs & ratings
└───────────────────────────────────────────┘
