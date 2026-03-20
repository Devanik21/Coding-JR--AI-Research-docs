# Coding Jr — AI Research Docs

![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github)
![Institute](https://img.shields.io/badge/NIT_Agartala-ECE-navy?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![Stars](https://img.shields.io/github/stars/Devanik21/Coding-JR--AI-Research-docs?style=flat-square&color=yellow)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

> A structured, technically rigorous collection of research documents on the mathematics, mechanics, and engineering of Large Language Models — written by a student who believes that understanding the internals is the only honest way to use any tool.

**Topics:** `transformer-architecture` · `fine-tuning` · `peft` · `rlhf` · `dpo` · `rag` · `vector-embeddings` · `ann-algorithms` · `prompt-compression` · `context-window` · `generative-ai` · `nlp-research`

---

## Overview

This repository is a research-oriented knowledge base covering five tightly interconnected areas in modern LLM engineering. The documents go beyond surface-level intuition and engage directly with the mathematical structures, algorithmic tradeoffs, and architectural decisions that define the field. The work originated as Week 2 curriculum notes and has since expanded into standalone reference documents, each built to be independently meaningful.

The central thesis threading all documents: **LLMs are function approximators bounded by architectural priors, training distributions, and finite context. Every technique in this collection is, at its core, an attempt to work around one of those three constraints.**

---

## Document Index

### 1. Understanding and Improving LLMs for Coding Tasks

Foundational entry into the corpus. Covers the Transformer architecture (Vaswani et al., 2017) from first principles: the self-attention mechanism, positional encoding, encoder-decoder decomposition, and the autoregressive training objective. The key operation — scaled dot-product attention — is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$, and the $\sqrt{d_k}$ scaling prevents dot-product magnitude growth that would push softmax into vanishing-gradient regimes. The document grounds this in the coding domain: tokenization of source code, the BERT/GPT architectural split (encoder-only for understanding, decoder-only for generation), and the cross-entropy training loss $\mathcal{L} = -\sum_i y_i \log \hat{y}_i$. Practical limitations specific to code — shallow structural understanding, hallucinated APIs, poor multi-file reasoning — are analyzed alongside improvement directions: contrastive learning on correct/buggy pairs, neuro-symbolic integration, and AST/CFG-aware generation.

---

### 2. Fitting Large Codebases into LLMs: A Comprehensive Deep Dive

The binding constraint in applied LLM engineering is the context window $C$ (in tokens). For a codebase of size $|X|$ tokens, the condition $|X| \gg C$ holds for virtually every non-trivial project. This document maps the full solution space for that constraint.

**RAG for Code** is the primary framework: (1) AST-based chunking via Tree-sitter for language-aware semantic units rather than arbitrary line boundaries, (2) code embedding $f: \mathcal{X}_{\text{code}} \to \mathbb{R}^d$ using code-specific models (VoyageCode3, CodeSage, Nomic Embed Code) that capture structural similarity better than general-purpose embedders, (3) ANN-indexed vector store, (4) top-$k$ similarity retrieval at query time, (5) grounded generation. Chunking granularity is a non-trivial hyperparameter: chunks too small lose semantic context, too large dilute embedding specificity.

Beyond RAG: hierarchical summarization (file → class → function abstractions as a "table of contents"), knowledge graph construction over call graphs for dependency-aware retrieval, agentic planner/editor role separation, and the hybrid RAG + fine-tuning configuration that delivers internalized style knowledge alongside real-time fact retrieval.

---

### 3. How Fine-Tuning Works in LLMs: A Deep Dive into Different Approaches

A systematic taxonomy of 28+ fine-tuning methods across four families: full fine-tuning, PEFT, instruction alignment, and specialized approaches.

**Full SFT** updates all parameters $\theta$ via AdamW, optimizing $\mathcal{L}_{\text{CE}} = -\sum_i y_i \log p_\theta(\hat{y}_i)$ at a lower learning rate than pre-training to avoid catastrophic forgetting. Storage cost: one full model copy per task — prohibitive at scale.

**LoRA** (Hu et al., 2021) is the pivotal PEFT contribution. For a frozen weight matrix $W_0 \in \mathbb{R}^{d \times k}$, the modified forward pass is:

$$h = W_0 x + \Delta W x = W_0 x + BAx$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, rank $r \ll \min(d, k)$. Trainable parameters drop from $dk$ to $r(d+k)$, typically 0.01–1% of the base model. At inference, $W' = W_0 + BA$ merges cleanly — zero added latency, unlike adapter-based methods.

**QLoRA** extends LoRA by quantizing $W_0$ to 4-bit NormalFloat (NF4), information-theoretically optimal for normally distributed weights, enabling 65B parameter models on a single 24 GB GPU. Double quantization further compresses the quantization constants.

**RLHF** trains reward model $r_\phi(x, y) \in \mathbb{R}$ on human preference pairs $(y_w \succ y_l \mid x)$, then optimizes the LLM policy $\pi_\theta$ via PPO with a KL penalty: $\mathcal{L} = \mathbb{E}[r_\phi(x,y)] - \beta\,\text{KL}[\pi_\theta \| \pi_\text{ref}]$. **DPO** eliminates the reward model entirely, recasting alignment as a classification loss directly on preference pairs. **KTO** introduces an asymmetric loss informed by Kahneman-Tversky prospect theory — penalizing dispreferred generations more heavily than rewarding preferred ones.

Other methods surveyed include: IA3 (learned activation scaling vectors), prefix-tuning (trainable key-value prefix per attention layer), P-tuning (LSTM-generated soft prompts), Houlsby/Pfeiffer adapters (bottleneck modules at residual connections), BitFit (bias-only, < 0.1% parameters), UniPELT (learned routing across PEFT methods), SVD-based updates, diff-pruning, EWC continual learning, multi-task learning, knowledge distillation, contrastive fine-tuning, RAFT, MoE fine-tuning, QAT, self-correction loops, and synthetic data via Self-Instruct.

---

### 4. Retrieval Augmented Generation (RAG) in Generative AI

Frames RAG as the principled solution to three structural failure modes of standalone LLMs: hallucination (the model is a next-token predictor, not a fact store), knowledge cutoff (static weights encode a historical snapshot), and niche knowledge gaps (long-tail information is underrepresented in pre-training).

Full pipeline: (1) **indexing** — chunking documents, embedding each chunk $c_i \mapsto \mathbf{v}_i \in \mathbb{R}^d$, storing in a vector database; (2) **retrieval** — embedding query $q \mapsto \mathbf{v}_q$, computing cosine similarity $\frac{\mathbf{v}_q \cdot \mathbf{v}_i}{\|\mathbf{v}_q\|\|\mathbf{v}_i\|}$, returning top-$k$ chunks; (3) **augmented generation** — constructing `[retrieved context] + [query]` and generating a grounded response. The document explains why vector databases differ fundamentally from relational databases (approximate semantic search vs. exact structured queries) and quantifies practical advantages: reduced hallucination, current information without retraining, domain specificity from curated knowledge bases, and answer traceability to source passages.

---

### 5. Vectorized Memory in Large Language Models: A Comprehensive Analysis

The most mathematically dense document. Treats vectorized memory as a formal system and dissects each component from first principles.

**Embeddings** are defined as $E: \mathcal{X} \to \mathbb{R}^d$, mapping arbitrary objects (text, images, audio, structured records) to a continuous space where geometric proximity encodes semantic relatedness. Traces the evolution from static Word2Vec/GloVe (single context-independent vector per word) to contextualized BERT-family embeddings where the representation of $t_i$ in sequence $(t_1, \ldots, t_n)$ depends on all other tokens via self-attention. Pooling strategies (CLS token, mean pooling, max pooling) for sentence-level embeddings are analyzed.

**Similarity metrics**: Cosine similarity $\frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|\|\mathbf{B}\|}$ is preferred for high-dimensional embeddings because it is magnitude-invariant. Euclidean distance $\sqrt{\sum_{i=1}^d (A_i - B_i)^2}$ is sensitive to vector norm and suffers from the **curse of dimensionality**: as $d \to \infty$, the ratio of maximum to minimum pairwise distances converges to 1, making all points appear equidistant.

**ANN algorithms**: Exact search is $O(Nd)$ — infeasible at $N \sim 10^9$, $d \sim 10^3$.

- **LSH**: hash $h(\mathbf{v}) = \text{sign}(\mathbf{w} \cdot \mathbf{v})$ for random normal $\mathbf{w}$; geometrically close vectors collide with high probability. Multiple hash tables trade storage for recall.
- **HNSW**: multi-layer graph with sparse long-range edges in upper layers (fast coarse navigation) and dense short-range edges in lower layers (local refinement). Empirical search complexity $O(\log N)$, best-in-class recall/latency.
- **IVF**: K-means partitions into $n_\text{centroids}$ Voronoi cells; search targets the $m$ nearest cells, reducing comparisons from $N$ to $\sim N/n_\text{centroids}$. Product Quantization (PQ) further compresses residual vectors within cells.

Applications: long-term conversational memory (retrieving relevant past dialogue turns), personalization (user-specific embedding stores), knowledge graph augmentation (embedding entities and relations for hybrid structured + semantic retrieval).

---

### 6. Text-to-Prompt Compression in LLMs

Frames prompt compression as a signal processing problem: input text $x$ contains signal $s$ (semantically load-bearing content) and noise $n$ (redundant, filler, tangential tokens). The goal is $C: x \mapsto x'$ where $|x'| = \rho|x|$ for $\rho < 1$ and task-relevant mutual information is maximized. Four motivations: cost (priced per token), context budget (fixed window $C$), latency (fewer tokens, shorter prefill), and signal-to-noise (attention sharpened over relevant content). Techniques: extractive summarization by TF-IDF or TextRank sentence scoring, abstractive re-generation, and keyphrase extraction as extreme compression.

---

## Recurring Mathematical Structures

| Object | Definition | Appears In |
|---|---|---|
| Context constraint | $\|x\|_{\text{tokens}} \leq C$ | Docs 1, 2, 3, 6 |
| Embedding map | $E: \mathcal{X} \to \mathbb{R}^d$ | Docs 2, 4, 5 |
| Cosine similarity | $(\mathbf{u} \cdot \mathbf{v}) / (\|\mathbf{u}\|\|\mathbf{v}\|)$ | Docs 4, 5 |
| Scaled attention | $\text{softmax}(QK^\top/\sqrt{d_k}) \cdot V$ | Docs 1, 5 |
| LoRA update | $\Delta W = BA,\; r \ll \min(d,k)$ | Docs 2, 3 |
| RLHF objective | $\mathbb{E}[r_\phi] - \beta\,\text{KL}[\pi_\theta \| \pi_\text{ref}]$ | Doc 3 |
| ANN complexity | $O(Nd) \to O(d \log N)$ | Docs 4, 5 |
| Compression ratio | $\|x'\| = \rho\|x\|,\; \rho < 1$ | Doc 6 |

---

## Repository Structure

```
Coding-JR--AI-Research-docs/
├── README.md
├── LICENSE
├── Week 2 _ Understanding and Improving LLMs for Coding Tasks.docx
├── Fitting Large Codebases into LLMs_ A Comprehensive Deep Dive.docx
├── How Fine-Tuning Works in LLMs_ A Deep Dive into different Approaches.docx
├── Retrieval Augmented Generation (RAG) in Generative AI_ A Simple Guide.docx
├── Vectorized Memory in Large Language Models_ A Comprehensive Analysis.docx
└── Text-to-Prompt Compression in LLMs_ Explanation.docx
```

---

## Proposed Research Directions — First Principles

These are not extensions of existing literature. They are derived by identifying the structural gaps the documents themselves expose, then proposing the minimal mechanism that addresses each gap. No borrowed framing.

---

### I. Gradient-Weighted Semantic Chunking (GWSC)

**The gap.** RAG chunking is syntactic (fixed windows, sentence boundaries) or structural (AST node types). Neither is aligned with what the model uses. A 300-token docstring may be semantically vacuous for a given retrieval task; a 5-line configuration block may be load-bearing. Chunk boundaries are chosen with no model signal.

**The idea.** During a calibration pass over a representative query set, compute the saliency of each token's hidden state with respect to the task loss: $g_i = \|\nabla_{h_i} \mathcal{L}\|_2$. Aggregate by AST node: $G_\text{node} = \frac{1}{|\text{node}|}\sum_{i \in \text{node}} g_i$. Place chunk boundaries at minima of $G$ — the model's own measure of semantic boundary.

**Why it matters.** Chunk boundaries become a learned function of model attention rather than an engineering heuristic. The resulting chunks are units of maximal model-relevant coherence by construction.

**Open questions.** Gradient computation at codebase scale is expensive; Fisher information or influence function approximations are needed. Saliency is query-distribution-dependent — multi-query averaging may be required.

---

### II. Topological Embedding Regularization (TER)

**The gap.** Embedding models are trained to place semantically similar objects nearby in $\mathbb{R}^d$, typically via contrastive loss. But the global topology of the learned manifold is uncontrolled: semantically distinct clusters may be entangled, a single concept may fragment into disconnected regions, and ANN retrieval — which operates on local distance — is blind to this global structure.

**The idea.** Apply persistent homology as a training regularizer. Construct the Vietoris-Rips filtration over a mini-batch embedding and compute persistence diagrams $\text{PD}_k$ for homological features at each dimension $k$ (connected components at $k=0$, loops at $k=1$, voids at $k=2$). Add a loss term $\lambda \cdot \mathcal{L}_\text{topo}$ penalizing long-lived $k \geq 1$ features that span semantically unrelated regions — topological noise without semantic grounding.

**Why it matters.** The topology of the embedding space directly determines ANN behavior. A topologically coherent manifold — well-separated clusters, concept-compact regions — improves top-$k$ recall at every $k$ without modifying the retrieval algorithm.

**Open questions.** Exact persistent homology is $O(n^3)$; batched approximations via Ripser on mini-batch subgraphs are needed. Defining "semantically unrelated" loops requires a grounding criterion — candidate: loops that cross known class or topic boundaries.

---

### III. Adaptive Rank Allocation in LoRA via Spectral Sensitivity (ARAS)

**The gap.** LoRA applies a uniform rank $r$ to all weight matrices. But attention query-key matrices in early layers, which perform broad syntactic matching, have structurally different intrinsic dimensionality than value-output matrices in later layers, which perform semantic composition. Uniform rank is a category error.

**The idea.** Before fine-tuning, compute $W_0 = U\Sigma V^\top$ for each weight matrix. Effective rank: $r^* = \arg\min_r \|\Sigma - \Sigma_r\|_F / \|\Sigma\|_F < \epsilon$. Also compute per-matrix gradient sensitivity $s_l = \|\nabla_{W_0^{(l)}} \mathcal{L}\|_F$ over a calibration batch. Allocate rank budget proportionally: $r_l \propto s_l$, subject to global budget $\sum_l r_l = R_\text{total}$.

**Why it matters.** Under a fixed parameter budget, ARAS concentrates expressiveness where the task gradient is largest. High-sensitivity matrices receive high rank; already-aligned matrices receive near-zero rank. Better per-parameter fine-tuning efficiency with no increase in total trainable parameters.

**Open questions.** Sensitivity estimation requires a forward-backward pass before training begins; Gauss-Newton approximations can reduce this cost. Optimal $\epsilon$ for effective rank estimation is task-dependent.

---

### IV. Preference Topology Learning for Alignment (PTL)

**The gap.** DPO and KTO model human preferences as pairwise scalars: $(y_w \succ y_l)$. But human preferences are multi-dimensional — helpfulness, harmlessness, honesty, and conciseness are distinct axes that can be orthogonal or in conflict. Collapsing them to a scalar reward discards this geometry and can satisfy one axis at the cost of another.

**The idea.** Embed responses in a shared semantic space and learn a preference **manifold** rather than a scalar. For human-annotated pairs along multiple axes, fit a Riemannian metric $g_{ij}(y)$ over the response embedding space such that geodesic distance reflects preference distance along each axis simultaneously. Alignment training then moves $\pi_\theta$ toward the Pareto frontier of the learned manifold — not a single scalar maximum.

**Why it matters.** Multi-objective alignment is a constrained optimization on a manifold, not a scalar maximization. Explicit geometry enables principled navigation: a safety-critical deployment targets one region of the manifold; a creativity-first application targets another — same base model, same preference geometry, different operating point.

**Open questions.** Learning a Riemannian metric from discrete pairwise annotations requires careful interpolation. Multi-axis annotation collection methodology and the intrinsic dimensionality of human preference space are both open.

---

### V. Compression-Retrieval Duality for Optimal Context Allocation (CRD)

**The gap.** Prompt compression and RAG are treated as separate techniques. Compression reduces what is already in context; RAG adds new content. Both operate on the same context budget $C$. In practice they are in tension: aggressive compression may discard content that retrieval would confirm as relevant, while retrieval may bring in content that compression would have removed. Neither method is aware of the other.

**The idea.** Formalize joint optimization: given query $q$, document set $\mathcal{D}$, and budget $C$, find content $x^* = \arg\max_{x \subseteq \mathcal{D}} I(x; q)$ subject to $\sum_i \rho_i |c_i|_\text{tokens} \leq C$, where $\rho_i \in (0, 1]$ is the compression ratio assigned to chunk $c_i$. Solve with a learned two-stage policy: a retriever scores chunks by relevance, a compressor allocates budget proportionally to chunk-level conditional mutual information $I(c_i; q)$ — high-relevance chunks kept near full, low-relevance chunks heavily compressed or discarded.

**Why it matters.** The context window is a channel with fixed capacity. Current practice (retrieve top-$k$ uncompressed, concatenate) is uniformly suboptimal. CRD treats context allocation as an information-theoretic problem and optimizes it end-to-end.

**Open questions.** Estimating $I(c_i; q)$ without ground truth requires a proxy — candidate: attention weight assigned to $c_i$ by a lightweight model. Training the joint policy requires a differentiable compression objective.

---

## References

- Vaswani et al. (2017). *Attention is All You Need.* NeurIPS.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* arXiv:1810.04805.
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314.
- Rafailov et al. (2023). *Direct Preference Optimization.* arXiv:2305.18290.
- Ethayarajh et al. (2023). *KTO: Model Alignment as Prospect Theoretic Optimization.* arXiv:2402.01306.
- Malkov & Yashunin (2018). *Efficient ANN Search Using HNSW Graphs.* IEEE TPAMI.
- Jégou et al. (2010). *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI.
- Reimers & Gurevych (2019). *Sentence-BERT.* arXiv:1908.10084.
- Indyk & Motwani (1998). *Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality.* STOC.

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

*Written with the conviction that the goal of studying a field is to find the questions it has not asked yet.*
