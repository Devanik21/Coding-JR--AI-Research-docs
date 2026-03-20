# Coding Jr — AI Research Docs

![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![Stars](https://img.shields.io/github/stars/Devanik21/Coding-JR--AI-Research-docs?style=flat-square&color=yellow)
![Forks](https://img.shields.io/github/forks/Devanik21/Coding-JR--AI-Research-docs?style=flat-square&color=blue)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

> A curated collection of deep-dive research documents on the mathematics, mechanics, and engineering of Large Language Models — written for developers who want more than surface-level intuition.

**Topics:** `llm` · `fine-tuning` · `rag` · `vector-embeddings` · `prompt-compression` · `transformer-architecture` · `generative-ai` · `nlp` · `research-notes`

---

## Overview

This repository is a focused, technically detailed knowledge base covering five interconnected areas in modern LLM engineering. Each document goes beyond introductory explanations — the goal is to give the reader a working mental model grounded in mathematics, architecture, and practical tradeoffs.

The documents emerged from a structured research and study process (Week 2 of a coding/AI curriculum), but are written to stand alone as reference material for anyone working in or studying applied AI.

---

## Document Index

### 1. Understanding and Improving LLMs for Coding Tasks
An introduction to the Transformer architecture — from the original *Attention is All You Need* (Vaswani et al., 2017) — through tokenization, embedding spaces, and the encoder/decoder distinction (BERT vs. GPT families). Covers how LLMs are adapted for programming languages and the tooling landscape (Copilot, Codex, CodeWhisperer, Gemini).

### 2. Fitting Large Codebases into LLMs: A Comprehensive Deep Dive
Addresses the core tension between finite context windows (measured in tokens, typically $n \in [10^4, 10^6]$) and the scale of real-world codebases (millions of lines, thousands of interdependent files). Explores chunking strategies, dependency-aware retrieval, and architectural patterns that allow LLMs to reason about code that vastly exceeds their context budget.

### 3. How Fine-Tuning Works in LLMs: A Deep Dive into Different Approaches
A systematic survey of 20+ fine-tuning methods, organized from full fine-tuning (all parameters $\theta$ updated, maximum expressiveness, maximum compute) through parameter-efficient methods. Covers Supervised Fine-Tuning (SFT), RLHF, LoRA, QLoRA, prefix tuning, prompt tuning, and adapter layers — with discussion of when each approach is appropriate and what it trades off.

### 4. Retrieval Augmented Generation (RAG) in Generative AI
Explains why retrieval is a principled solution to hallucination, knowledge cut-offs, and niche-domain gaps. Walks through the full RAG pipeline: query encoding → ANN search over a vector store → context injection → grounded generation. Suitable as both a conceptual introduction and a technical reference.

### 5. Vectorized Memory in Large Language Models: A Comprehensive Analysis
The most mathematically rigorous document in the collection. Covers dense vector embeddings, similarity metrics (cosine similarity, dot product, Euclidean distance), and approximate nearest-neighbour (ANN) algorithms (HNSW, IVF, FAISS). Frames vectorized memory as a solution to the static-knowledge and context-window limitations inherent in pre-trained LLMs.

### 6. Text-to-Prompt Compression in LLMs
Examines prompt compression as an efficiency lever — reducing token count (and therefore cost and latency) while preserving semantic content. Covers the four core benefits: cost reduction, context extension, faster inference, and improved signal-to-noise ratio. Structured as a technical explainer / presentation script.

---

## Mathematical Themes

Several formal ideas recur across documents and are worth naming explicitly:

- **Context window constraint:** $|\text{tokens}(x)| \leq C$, where $C$ is the model's maximum context length. The entire field of RAG, compression, and codebase-fitting exists to navigate this constraint.
- **Embedding function:** $f: \mathcal{X} \to \mathbb{R}^d$, mapping discrete tokens or documents into a continuous vector space where geometric proximity encodes semantic similarity.
- **Cosine similarity:** $\text{sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$, the standard retrieval metric for comparing query and document embeddings.
- **Fine-tuning objective:** $\min_{\theta'} \mathcal{L}(\theta_{\text{pretrained}} + \Delta\theta)$ — updating a small parameter delta $\Delta\theta$ rather than the full weight matrix, as in LoRA.

---

## Repository Structure

```
Coding-JR--AI-Research-docs/
├── README.md
├── LICENSE
├── Fitting Large Codebases into LLMs_ A Comprehensive Deep Dive.docx
├── How Fine-Tuning Works in LLMs_ A Deep Dive into different Approaches.docx
├── Retrieval Augmented Generation (RAG) in Generative AI_ A Simple Guide.docx
├── Text-to-Prompt Compression in LLMs_ Explanation.docx
├── Vectorized Memory in Large Language Models_ A Comprehensive Analysis.docx
└── Week 2 _ Understanding and Improving Large Language Models (LLMs) for Coding Tasks.docx
```

---

## Who This Is For

These docs are best suited for:

- **Developers** building LLM-powered applications who want to understand what's happening under the hood.
- **Students** in AI/ML programmes looking for well-structured reference material beyond lecture slides.
- **Researchers** who want a concise survey of a topic before going deeper into primary literature.

Some familiarity with linear algebra and basic ML concepts (loss functions, gradient descent, neural network layers) is assumed, particularly for the vectorized memory and fine-tuning documents.

---

## Roadmap

- [ ] Add mathematical appendix consolidating notation across all documents
- [ ] Supplement each doc with a code walkthrough (Python / Jupyter)
- [ ] Expand the RAG guide with an end-to-end implementation using LangChain or LlamaIndex
- [ ] Add a comparative study of ANN libraries: FAISS vs. Chroma vs. Pinecone vs. Weaviate
- [ ] Convert select documents to interactive notebooks

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

Open source under the [MIT License](LICENSE).

---

*Built with intellectual curiosity and a conviction that understanding the mathematics makes you a better engineer.*
