# ⚖️ Autonomous Multi-Agent Debate Engine

An advanced, responsive multi-agent orchestration platform that enables multi-turn, interactive semantic argumentation between human users and advanced Large Language Models (LLMs). The engine tracks conversations dynamically over 3 complete rebuttal cycles and utilizes a dedicated third-party LLM tribunal to evaluate performance scoring parameters.

## 🚀 Core Engineering Highlights

* **Dynamic RAG Context Synthesizer:** Incorporates an in-memory ChromaDB vector store paired with local `all-MiniLM-L6-v2` dense vector representations. Features a self-healing layer that dynamically spins up factual document nodes if a user provides an out-of-domain topic.
* **Conversational State Tracking Machine:** Uses state-tracking modules (`st.session_state`) to maintain conversational context over a multi-turn paradigm against traditionally stateless foundational API endpoints.
* **Deterministic LLM-as-a-Judge Evaluation:** Features a tailored, low-temperature evaluation agent ($T=0.1$) that enforces strict analytical scrutiny over complete chat logs, delivering structured evaluations in validated JSON formats.
* **High-Contrast Professional UI Framework:** Built natively using custom CSS layout overrides, strict device proportions, and crisp, highly readable text elements.

---

## 🛠️ Architecture and Stack Specification

* **Interface:** Streamlit Frame (Python)
* **Agent Framework:** LangChain Core Engine
* **Vector Model:** HuggingFace Embedding Space (`all-MiniLM-L6-v2`, 384 dimensions)
* **Vector Index Matrix:** Chroma Vector DB (In-Memory Implementation)
* **Foundation Models:** Google Gemini Inference API (`gemini-2.5-flash`)

---

## 🏃‍♂️ How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/srivallisv/AI-Debate-Engine.git](https://github.com/srivallisv/AI-Debate-Engine.git)
cd AI-Debate-Engine
