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
