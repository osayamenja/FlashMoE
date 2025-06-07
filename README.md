<p align="center">
  <img src="logow.png" alt="Kleos Conceptual Overview" width="200"/>
</p>

<p align="center"><i>Complete, efficient GPU residency for Machine Learning workloads</i></p>

---

## 🌌 Kleos: GPU-Resident Runtime for ML

**Kleos** is an ongoing research project exploring the design of a GPU-native operating system for distributed machine learning workloads.  
The goal is to eliminate CPU bottlenecks by fusing scheduling, communication, and compute **directly on the GPU** using lightweight, persistent runtime primitives.

Kleos targets irregular and sparse workloads such as **Mixture-of-Experts (MoE)**, where conventional bulk-synchronous, CPU-driven orchestration becomes a limiting factor.

> This repository represents a *very* early-stage release of Kleos infrastructure.

---

## 🗞️ News

- **June 2025** — ⚡️Excited to release **FlashDMoE**, a fused GPU kernel for distributed MoE execution.  
  ➤ See [`this README`](./csrc/include/moe/README.MD) for details, benchmarks, and usage.

---

## ⚖️ License

This project is licensed under the BSD 3-Clause License. See [`LICENSE`](./LICENSE) for full terms.
