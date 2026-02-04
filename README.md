
# Hybrid Meta-Heuristics: LLM-Enhanced Particle Swarm Optimization

**Course:** Metaheuristic Optimization (Fall 2025)  
**Supervisor:** Dr. Ebadzadeh  
**Project Type:** Research Implementation & Enhancement  
**Domain:** Evolutionary Computation, Large Language Models (LLMs), Hyperparameter Optimization (HPO)

---

## 📌 Project Overview

This project implements a novel **Hybrid Meta-Heuristic framework** that integrates **Large Language Models (LLMs)** with **Particle Swarm Optimization (PSO)**. The goal is to optimize hyperparameters for Deep Learning models (CNNs and LSTMs) more efficiently than traditional methods.

The project is divided into two phases:
1.  **Phase 1 (Baseline & Validation):** Comparison of Simple PSO, Vanilla Improved PSO (Adaptive Inertia + Niching), and Basic LLM-Enhanced PSO.
2.  **Phase 2 (Advanced Enhancements):** Implementation of LLM as a **Meta-Controller** (dynamic $c_1/c_2$ tuning), expansion to **5D Search Spaces**, and comparative analysis of advanced **Inertia Weight Strategies**.

### Key Innovations
*   **LLM as a Meta-Optimizer:** Unlike standard approaches where LLMs just suggest coordinates, our Phase 2 engine uses the LLM to analyze particle velocity history and dynamically tune the swarm's cognitive ($c_1$) and social ($c_2$) coefficients.
*   **RAG-based Optimization Memory:** A memory module tracks the history of particle states, allowing the LLM to use "Chain-of-Thought" reasoning to suggest improvements.
*   **Multi-Strategy Inertia:** Implementation of success-rate-based and rank-based inertia weights to balance exploration and exploitation.

---

## 🧠 Methodology & Architectures

### 1. Phase 1: Baseline Comparisons
In this phase, we established a baseline by comparing three distinct architectures:
*   **Simple PSO:** Standard implementation with fixed inertia and coefficients. Often suffers from premature convergence.
*   **Vanilla Improved PSO:** Incorporates **Adaptive Inertia Weight (Linearly Decreasing)** and a **"Lone Wolf" Particle** (Cognitive-only) that ignores the global best to maintain diversity (Niching technique).
*   **LLM-Enhanced PSO (Basic):** Periodically pauses the swarm to query an LLM (GPT-5.1) for better potential coordinates based on the current fitness landscape.

### 2. Phase 2: Advanced Concepts

#### Task A: LLM-Controlled Dynamics ($c_1, c_2$ & RAG)
Instead of static coefficients, the LLM analyzes the swarm's behavior (velocity magnitudes and convergence rate).
*   **Context:** The LLM receives a history of particle positions and velocities.
*   **Action:** It dynamically adjusts:
    *   **$c_1$ (Cognitive):** Increased if particles are exploring too little locally.
    *   **$c_2$ (Social):** Decreased if the swarm is converging too fast to a local optimum.
    *   **Position Reset:** Resets the worst-performing particles to LLM-suggested "promising regions."

#### Task B: High-Dimensional Search (5D)
Optimization was expanded from 2 dimensions (Layers, Neurons) to 5 dimensions to find robust architectures:
1.  **Layers:** Network Depth ($1-6$).
2.  **Neurons/Filters:** Network Width ($16-256$).
3.  **Learning Rate:** Step size ($1e-4 - 1e-2$).
4.  **Dropout:** Regularization ($0.0 - 0.5$).
5.  **Epochs:** Training duration per particle.

#### Task C: Inertia Weight ($w$) Strategies
We implemented and compared 5 strategies based on *Nickabadi et al.*:
1.  **Linear Decreasing ($w_{dec}$):** Standard exploration $\to$ exploitation.
2.  **Linear Increasing ($w_{inc}$):** Tests the inverse hypothesis.
3.  **Random ($w_{rand}$):** Stochastic behavior.
4.  **Success-Rate Based (AIWPSO):** $w$ adapts based on how many particles improved their personal bests ($P_{best}$) in the last iteration. High success = High $w$ (Explore).
5.  **Rank-Based:** Each particle gets a unique $w$ based on its fitness rank (Better particles exploit, worse particles explore).

---

## 🧪 Experimental Setup

### Classification Task
*   **Dataset:** Waste Classification Data (Organic vs. Recyclable).
*   **Model:** Dynamic CNN (Variable Convolutional layers and filters).
*   **Objective:** Maximize **Accuracy**.
*   **Constraint:** Training limited to 25% of data for rapid evolutionary evaluation.

### Regression Task
*   **Dataset:** Beijing PM2.5 Data (UCI Machine Learning Repository).
*   **Model:** Dynamic LSTM (Variable LSTM layers and hidden units).
*   **Objective:** Minimize **RMSE** (Root Mean Square Error).
*   **Input:** Sequential sensor data (DEWP, TEMP, PRES, Iws, Is, Ir) with a window size of 24.

---

## 📊 Comparative Analysis of Architectures

*Note: This section qualitatively compares the implemented methods.*

| Feature | Simple PSO | Vanilla Improved (Phase 1) | LLM-Enhanced (Phase 1) | LLM-Meta-Control (Phase 2) |
| :--- | :--- | :--- | :--- | :--- |
| **Inertia Strategy** | Fixed (0.7) | Linear Decrease | Linear Decrease | **Dynamic / Success-Based** |
| **Diversity Mech.** | None | Lone Wolf (Niching) | LLM Suggestions | **Chain-of-Thought Reasoning** |
| **Parameter Control**| Static | Static | Static | **Dynamic ($c_1, c_2$ by LLM)** |
| **Search Dims** | 2D | 2D | 2D | **5D (incl. Hyperparams)** |
| **Convergence** | Fast (Risk of Local Optima) | Balanced | Robust | **Highly Robust** |

**Observations from Code Implementation:**
1.  **Memory Module:** The `OptimizationMemory` class acts as a textual buffer, converting numerical vectors into a prompt that the LLM can "read," enabling it to understand trends like "increasing layers is worsening the RMSE."
2.  **Adaptive Strategies:** The `Task3` code demonstrates that **Success-Rate** and **Rank-Based** strategies provide a more granular control over particle momentum compared to simple linear decay.
3.  **Regularization:** In the 5D search, the inclusion of **Dropout** allows the optimizer to select larger models (more layers) without overfitting, which was not possible in the 2D search space.

---

## 🔧 Installation and Usage

### Prerequisites
*   Python 3.8+
*   PyTorch (CUDA recommended)
*   AvalAI API Key (or OpenAI compatible endpoint)

```bash
pip install torch torchvision pandas numpy scikit-learn requests
```

---

## 📜 References

1.  **Primary Baseline:** S. Hameed, et al., *"Large Language Model Enhanced Particle Swarm Optimization for Hyperparameter Tuning for Deep Learning Models,"* IEEE Open Journal of the Computer Society, 2025.
2.  **Adaptive Inertia:** A. Nickabadi, M. M. Ebadzadeh, and R. Safabakhsh, *"A novel particle swarm optimization algorithm with adaptive inertia weight,"* Applied Soft Computing, 2011.
3.  **DNPSO (Niching):** A. Nickabadi, et al., *"DNPSO: A Dynamic Niching Particle Swarm Optimizer for multi-modal optimization,"* IEEE CEC, 2008.

---

*Implementation by Mohsen Hami | Fall 2025*
```
