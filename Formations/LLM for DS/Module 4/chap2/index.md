---
layout: default
title: Retrieval Augmented Generation and Agents
---


# 🤖 Chapter 2: LLM Agents

## 2.1 🧠 Introduction to LLM Agents

LLM agents are autonomous systems powered by Large Language Models (LLMs) that can perform complex, multi-step tasks by combining reasoning, memory, planning, and tool use.

Unlike standard prompt-based LLM usage, agents behave more like decision-makers: they analyze a task, decide what to do, fetch the right information, take actions (often using tools), and generate coherent results — all iteratively and independently.


---

## 2.2 🧩 Core Components of an LLM Agent

- **🧠 LLM Controller**  
  The main reasoning unit. It plans, decides, and orchestrates the agent's flow using natural language.

- **🗺️ Planning Module**  
  Decomposes the user goal into sub-tasks or a sequence of actions.

- **🧠 Memory**  
  Stores context, past interactions, intermediate results, and world knowledge (short-term or long-term).

- **🔌 Tool Use**  
  Accesses external APIs, web search, calculators, databases, RAG retrievers, and more.

- **🗃️ Scratchpad / Workspace**  
  A structured place where intermediate steps and thoughts are tracked (like chain-of-thought reasoning or agent logs).

![image](https://github.com/user-attachments/assets/d2d7fd89-bd1c-4306-89d4-f2de05944cce)

---

## 2.3 🔄 Agent Workflow

1. **User Input**  
   The agent receives a task or query from the user.

2. **Planning**  
   The LLM controller (optionally with a planner) determines what steps are needed.

3. **Memory Lookup / Retrieval**  
   Accesses prior context or invokes a retriever (e.g., vector DB) to gather relevant info.

4. **Tool Calls / Actions**  
   Executes API calls, web searches, code execution, etc., as needed to complete steps.

5. **Intermediate Reasoning**  
   Updates the scratchpad or memory with progress toward the goal.

6. **Final Answer**  
   Generates and returns a complete, structured output.

7. **Learning / Update**  
   Optionally stores what was learned for future tasks.

---

## 2.4 🧠 Agentic RAG

**Agentic RAG** combines the structured retrieval pipeline of RAG with the autonomy and planning capabilities of LLM agents.

Instead of a static “retrieve-then-generate” approach, an agentic system can **reason about what to retrieve, how to retrieve it, and when to stop retrieving** — adapting dynamically based on the evolving task.

---

### 🔄 Key Differences from Classic RAG

| Classic RAG | Agentic RAG |
|-------------|-------------|
| 📥 One-shot query → top-k retrieval → LLM answers | 🤔 Agent decomposes query, decides retrieval strategy per subtask |
| 🔍 Static top-k chunk fetching | 🔁 Iterative, conditional, or multi-stage retrieval |
| 🧠 LLM is a passive consumer of context | 🎯 LLM actively drives retrieval and decides what info it needs |

---

### 🧩 Typical Agentic RAG Workflow

1. **Initial Query**  
   User asks a broad or multi-part question.

2. **Query Decomposition**  
   Agent breaks the question into steps or sub-questions.

3. **Step-by-Step Retrieval**  
   For each step, the agent dynamically selects:
   - Which retriever to use (dense, sparse, KG, etc.)
   - What metadata filters to apply
   - Whether to retrieve again based on quality

4. **Tool Use (Optional)**  
   May include table lookup, API calls, or document summarization.

5. **Answer Synthesis**  
   The agent combines intermediate results and generates a final grounded answer.

6. **Self-Critique**  
   Some agents re-evaluate their draft (e.g., using a second LLM pass) for correctness.


---

### ✅ Benefits

- 📈 Higher factual accuracy for multi-hop questions  
- 🔁 More efficient use of context window  
- 🤖 Better performance in open-domain QA or ambiguous prompts  
- 🧠 Closer to how a human researches and composes an answer

---

### 🚧 Challenges

- ⏱️ Latency: Multiple retrieval/generation steps  
- 💡 Evaluation: Harder to benchmark step-wise decisions  
- 🧪 Debugging: Complex reasoning paths require observability tools  
- 🧰 Requires tool and prompt orchestration infrastructure

---

> Agentic RAG is a natural evolution of classic RAG — empowering language models not just to *read*, but to *think, search, and decide* as part of the answer process.



## 2.4 🎯 Why Use LLM Agents?

- **🧠 Autonomy**  
  Agents make multiple decisions without constant user prompts.

- **🧩 Reasoning**  
  Can break down complex tasks into smaller steps and solve them iteratively.

- **🔗 Tool Use**  
  Extend beyond text generation: browse the web, execute code, access private databases, etc.

- **🗃️ Context Persistence**  
  Memory lets them learn from previous interactions or work across long workflows.

- **🚀 Scalability**  
  You can orchestrate multi-agent systems with specialization for complex pipelines.

---

## 2.5 🏗️ Architectures of LLM Agents

### 🔹 Single-Agent Systems  
One agent handles planning, decision-making, and tool use. Ideal for linear, well-scoped tasks (e.g., research assistant).
![image](https://github.com/user-attachments/assets/0034fec5-f3a4-45c6-86e7-57ab7880c11f)


### 🔸 Multi-Agent Systems  
Multiple agents with specialized roles (e.g., planner, retriever, coder, verifier). Agents can communicate via shared memory or message passing.

![image](https://github.com/user-attachments/assets/7718640e-2e32-4b7e-a116-58e4c9dc562e)


Example use cases:
- Autonomous research workflows
- Long-horizon customer support
- Developer assistants with internal tools

---

## 2.6 🛠️ Building LLM Agents

To implement LLM agents in practice, consider the following components:

- **🧱 Frameworks & Tools**
  - [OpenAI Agent](https://openai.github.io/openai-agents-python/)
  - [LangChain Agents](https://docs.langchain.com/docs/components/agents/)  
  - [LlamaIndex Agents](https://docs.llamaindex.ai/en/stable/examples/agents/)

- **🧠 Memory Solutions**  
  - LangChain memory classes  
  - Redis or vector stores (for persistent context)  
  - Episodic vs. semantic memory patterns

- **🔌 Toolkits to Integrate**  
  - Calculator  
  - Web Search  
  - Python REPL / Code interpreter  
  - Retrieval-Augmented Generation (RAG)  
  - SQL / API connectors

- **📏 Evaluation**  
  - Use task success rate, faithfulness, and human judgment  
  - Monitor reasoning steps, hallucination rate, tool failure rate

---

## 2.7 🔬 Limitations & Challenges

- ❌ Agents can hallucinate tools or steps
- 🐢 Latency: multiple steps = longer execution times
- 📉 Tool reliability: tool outputs must be validated
- 📊 Hard to evaluate: open-ended tasks don't always have clear metrics
- 🔄 Memory persistence: long-term memory is still experimental

---

## 2.8 📚 Further Reading

- 📖 [Prompting Guide: LLM Agents](https://www.promptingguide.ai/research/llm-agents)  
- 📘 [A Visual Guide to LLM Agents (Maarten Grootendorst)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents)  
- 📙 [Understanding Multi-Agent Architectures](https://medium.com/@pallavisinha12/understanding-llm-based-agents-and-their-multi-agent-architecture-299cf54ebae4)

---

## ✅ Summary

- LLM agents are autonomous, goal-driven systems powered by large language models  
- They integrate planning, memory, and tools to handle complex tasks  
- Frameworks like LangChain, CrewAI, and Autogen make them accessible  
- Evaluate and monitor carefully: agents are powerful but unpredictable  
- Future AI systems will increasingly rely on agent-based orchestration

---
> 📌 Key source: [Prompting Guide – LLM Agents](https://www.promptingguide.ai/research/llm-agents)



