<div align="center">

# 🧠 Self-Corrective RAG
### *Evaluation-Driven Retrieval-Augmented Generation with Autonomous Recovery & LangGraph Orchestration*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-Orchestration-6C63FF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Groq-LLM-F55036?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PostgreSQL-Neon-336791?style=for-the-badge&logo=postgresql&logoColor=white"/>
  <img src="https://img.shields.io/badge/pgvector-Vector%20Database-2F855A?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/RAGAS-Evaluation-7B42BC?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Docker-Deployment-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

### 🚀 A Production-Ready Self-Corrective Retrieval-Augmented Generation Framework that **evaluates**, **diagnoses**, and **recovers** from retrieval failures and hallucinations before delivering grounded, trustworthy AI responses.

<p align="center">

[🌐 Live API](https://self-healing-rag-api-v3.onrender.com) •
[📖 API Docs](https://self-healing-rag-api-v3.onrender.com/docs) •
[💻 GitHub Repository](https://github.com/shiwan-mangate/Self-Corrective-RAG)

</p>

</div>

---

# 📖 Overview

Large Language Models are only as reliable as the information they receive. Traditional Retrieval-Augmented Generation (RAG) systems retrieve relevant documents once, generate an answer, and immediately return it to the user—even when the retrieved context is incomplete, irrelevant, or hallucination-prone.

**Self-Corrective RAG** introduces an evaluation-driven architecture that continuously validates every generated response before it reaches the user. Instead of assuming retrieval succeeded, the system performs multiple quality checks, diagnoses failures, automatically executes recovery strategies, and only then returns the final answer.

The entire workflow is orchestrated using **LangGraph**, enabling cyclic execution, intelligent routing, retry management, and autonomous recovery while maintaining conversation memory and long-term knowledge improvement.

Unlike conventional RAG systems, this framework doesn't stop after generation—it actively **thinks, evaluates, corrects, and learns**.

---

# ✨ Why Self-Corrective RAG?

Traditional RAG pipelines generally follow a simple linear workflow:

```
User Query
      │
      ▼
Retrieve Documents
      │
      ▼
Generate Answer
      │
      ▼
Return Response
```

This approach works well only when retrieval is perfect.

In real-world scenarios, failures occur frequently:

- Retrieved chunks may be irrelevant.
- Important knowledge may be missing.
- The LLM may hallucinate unsupported facts.
- Follow-up questions lose conversational context.
- Out-of-domain queries often fail silently.

Rather than returning unreliable answers, **Self-Corrective RAG** introduces an autonomous recovery layer that continuously monitors answer quality and dynamically repairs failures before responding.

Its execution lifecycle looks like:

```
User Query
      │
      ▼
Memory
      │
      ▼
Retrieval
      │
      ▼
Generation
      │
      ▼
Evaluation
      │
      ▼
Self-Healing
      │
      ▼
Retry Guard
      │
      ▼
Final Response
```

This architecture enables the system to recover from poor retrieval, hallucinations, missing knowledge, and low-confidence generations automatically.

---

# 🌟 Key Highlights

- 🧠 LangGraph-based cyclic AI workflow
- 📚 Multi-format document ingestion pipeline
- 🔍 Semantic retrieval using Hugging Face embeddings
- ✍️ Intelligent query rewriting
- 📑 Context-aware reranking and filtering
- 💬 Persistent conversational memory
- 🛡️ Strict context-grounded generation
- 📖 Automatic citation extraction
- 🚨 Hallucination detection
- 📊 Confidence scoring
- 📈 RAGAS-based evaluation framework
- 🔄 Autonomous self-healing recovery
- 🌐 Web search fallback using Tavily
- 🧩 Knowledge-gap detection
- 📥 Continuous knowledge ingestion
- 🚦 Retry guard to prevent infinite execution loops
- ⚡ FastAPI production API
- 🐳 Docker & Render deployment
- 🧪 Comprehensive Unit, Integration, and End-to-End testing

---

# 🔥 What Makes This Different?

Most RAG implementations stop after generating an answer.

**Self-Corrective RAG** adds an entire autonomous reasoning layer on top of traditional retrieval.

Instead of asking:

> **"Can the model answer this question?"**

it asks:

- **Was the retrieved context sufficient?**
- **Is the answer grounded in evidence?**
- **Did the model hallucinate?**
- **How confident is the final response?**
- **Should the system rewrite the query?**
- **Should external knowledge be fetched?**
- **Can this failure improve future responses?**

This transforms a standard RAG pipeline into an intelligent system capable of continuously improving response quality.

---

# 🚀 Live Demo

| Service | Link |
|---------|------|
| 🌐 Live API | https://self-healing-rag-api-v3.onrender.com |
| 📖 Swagger UI | https://self-healing-rag-api-v3.onrender.com/docs |
| 💻 GitHub Repository | https://github.com/shiwan-mangate/Self-Corrective-RAG |

---

# 📚 Table of Contents

- [📖 Overview](#-overview)
- [✨ Why Self-Corrective RAG?](#-why-self-corrective-rag)
- [🌟 Key Highlights](#-key-highlights)
- [🔥 What Makes This Different?](#-what-makes-this-different)
- [🚀 Live Demo](#-live-demo)
- [🏗️ System Architecture](#️-system-architecture)
- [⚙️ Pipeline Architecture](#️-pipeline-architecture)
- [🚀 Features & Technology](#-features--technology)
- [🛠️ Getting Started](#️-getting-started)
- [📡 API & Performance](#-api--performance)
- [🧪 Testing & Roadmap](#-testing--roadmap)
- [📄 License & Author](#-license--author)

  # 🏗️ System Architecture

Self-Corrective RAG is designed as a **modular, evaluation-driven AI system** where every stage of the Retrieval-Augmented Generation (RAG) lifecycle is isolated into independent subsystems and orchestrated through **LangGraph**.

Unlike traditional RAG pipelines that immediately return an LLM-generated response after retrieval, this framework continuously **evaluates**, **diagnoses**, **recovers**, and **learns** before producing the final answer.

At the center of the architecture is a cyclic LangGraph workflow that enables intelligent routing, autonomous recovery, conversation memory, and continuous knowledge improvement.

---

# 🎯 Why Self-Corrective RAG?

Modern Retrieval-Augmented Generation (RAG) systems significantly reduce hallucinations by grounding LLMs on retrieved documents. However, they still assume that the retrieval stage always succeeds.

In practice, this assumption frequently breaks.

Real-world enterprise knowledge bases continuously evolve, documents become outdated, user queries are ambiguous, and retrieved context is often incomplete or irrelevant. Once retrieval fails, conventional RAG systems continue generating answers from insufficient evidence, producing hallucinations or low-confidence responses without recognizing their own mistakes.

Self-Corrective RAG addresses this limitation by introducing an **Evaluation-Driven Feedback Loop**.

Instead of trusting every generated response, the system continuously verifies response quality, identifies failure causes, executes targeted recovery strategies, and improves the knowledge base whenever missing information is detected.

The result is an intelligent RAG architecture capable of **reasoning about its own failures before responding to users.**

---

# Traditional RAG vs Self-Corrective RAG

| Traditional RAG | Self-Corrective RAG |
|-----------------|--------------------|
| Single-pass pipeline | Cyclic feedback architecture |
| Retrieves once | Can retrieve multiple times |
| No quality verification | Evaluates every generated response |
| Hallucinations may reach users | Hallucinations are detected before response |
| Cannot recover from poor retrieval | Automatic recovery strategies |
| Static knowledge base | Continuous knowledge improvement |
| No retry mechanism | Intelligent retry routing |
| No long-term learning | Knowledge-gap detection & ingestion |
| Minimal observability | Full execution tracing and metrics |
| Linear execution | LangGraph-based orchestration |

---

# 🏢 Overall Enterprise Architecture

<p align="center">

> **Replace this with your enterprise architecture diagram**

<img src="images/architecture.png" width="100%">

</p>

The architecture is organized into independent, loosely coupled layers, each responsible for a specific stage of the AI lifecycle.

Every subsystem follows the **Single Responsibility Principle**, making the project highly modular, testable, maintainable, and extensible.

---

# 🔄 Complete AI Lifecycle

The complete execution lifecycle consists of seven specialized AI subsystems coordinated through LangGraph.

```text
                     User Query
                         │
                         ▼
              Input Preparation Layer
                         │
                         ▼
                 Memory Subsystem
                         │
                         ▼
               Retrieval Subsystem
                         │
                         ▼
               Generation Subsystem
                         │
                         ▼
               Evaluation Subsystem
                         │
                         ▼
             Self-Healing Subsystem
                         │
                         ▼
                Retry Guard Router
               ┌─────────┴─────────┐
               │                   │
      Continue Recovery      Return Response
               │                   │
               ▼                   ▼
        Retrieval / Generation   Persist Memory
```

Instead of following a simple linear pipeline, the workflow forms a **closed feedback loop** where every generated response is validated before reaching the user.

---

# 🧱 Layered Architecture

The framework is divided into specialized layers that collectively implement an enterprise-grade Self-Corrective RAG pipeline.

| Layer | Responsibility |
|--------|----------------|
| **Layer 0** | API Layer (FastAPI, Request Validation, Dependency Injection) |
| **Layer 1** | Document Ingestion & Vectorization |
| **Layer 2** | Conversation Memory Management |
| **Layer 3** | Semantic Retrieval & Context Construction |
| **Layer 4** | Context-Grounded Response Generation |
| **Layer 5** | Response Evaluation & Quality Verification |
| **Layer 6** | Autonomous Self-Healing & Recovery |
| **Layer 7** | LangGraph Orchestration & Intelligent Routing |
| **Infrastructure** | PostgreSQL, pgvector, Hugging Face Embeddings, Docker, Render |

Each layer exposes a clean pipeline interface while hiding implementation details behind dependency injection, allowing components to evolve independently without affecting the rest of the system.

---

# ⚙️ End-to-End Workflow

The following illustrates how a user query travels through the entire system.

```text
┌──────────────────────────┐
│        User Query        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Input Preparation Layer  │
│ • Validation             │
│ • Normalization          │
│ • Execution Metadata     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Memory Pipeline          │
│ • Load Session           │
│ • Conversation History   │
│ • Context Summary        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Retrieval Pipeline       │
│ • Query Analysis         │
│ • Query Rewriting        │
│ • Semantic Search        │
│ • Filtering              │
│ • Reranking              │
│ • Context Builder        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Generation Pipeline      │
│ • Prompt Construction    │
│ • LLM Generation         │
│ • Citation Extraction    │
│ • Response Formatting    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Evaluation Pipeline      │
│ • Grounding Verification │
│ • Hallucination Check    │
│ • Confidence Score       │
│ • RAGAS Evaluation       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Self-Healing Pipeline    │
│ • Recovery Diagnosis     │
│ • Query Rewrite          │
│ • Web Search             │
│ • Context Merge          │
│ • Knowledge Gap          │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Retry Guard             │
│ • Prevent Infinite Loop │
│ • Route Next Step       │
└───────┬────────┬─────────┘
        │        │
        │Retry   │Success
        ▼        ▼
 Retrieval     Final Response
                  │
                  ▼
        Conversation Persistence
```

---

# 🧠 Architectural Principles

The architecture follows several software engineering principles that make it suitable for production AI systems.

- **Pipeline-Based Design** – Every subsystem has a single responsibility and exposes one public entry point.
- **Dependency Injection** – Components are loosely coupled and independently testable.
- **Evaluation-Driven Intelligence** – Responses are validated before being returned.
- **Autonomous Recovery** – The system attempts to repair failures without user intervention.
- **Continuous Learning** – Missing knowledge is identified and prepared for future ingestion.
- **Graph-Based Orchestration** – LangGraph coordinates execution, routing, retries, and recovery.
- **Observability First** – Every stage records latency, quality metrics, and execution metadata.
- **Production Ready** – Dockerized deployment with FastAPI, PostgreSQL, and Render support.

---

> **In the following section, we dive into each subsystem individually, exploring how the Ingestion, Memory, Retrieval, Generation, Evaluation, Self-Healing, and LangGraph pipelines work internally.**


