# Machine Learning Projects Portfolio

A collection of production-grade machine learning implementations built from first principles. Each project demonstrates deep understanding of core algorithms through clean, modular code, reproducible validation, and interview-ready documentation.

## Projects Overview

### 0. Real-Time Model Serving and Monitoring Pipeline
**Directory:** `real-time-model-serving-monitoring/`

A production-style MLOps project demonstrating real-time inference, model artifact versioning, request validation, prediction logging, latency telemetry, and feature-drift monitoring.

**Key Features:**
- FastAPI service with `/health`, `/predict`, `/batch_predict`, `/monitor/drift`, and `/monitor/latency`
- Reproducible scikit-learn training pipeline with generated model artifact, metadata, baseline feature statistics, and model card
- p50/p95 latency tracking, request logging, and PSI-based feature drift monitoring
- Dockerfile, example payloads, and automated tests covering API and monitoring paths
- Holdout metrics from reproducible run: Accuracy 94.7%, ROC-AUC 0.993, F1 0.958

**Technologies:** FastAPI, Docker, scikit-learn, Pydantic, pytest, model monitoring

**Use Cases:** MLOps, production ML, real-time inference, model monitoring, AI/ML engineer interviews

---

### 1. Diffusion Models from Scratch
**Directory:** `diffusion-models/`

A complete implementation of Denoising Diffusion Probabilistic Models (DDPM) for image generation.

**Key Features:**
- Forward and reverse diffusion processes with multiple noise schedules
- Custom UNet architecture with attention mechanisms
- DDPM and DDIM sampling algorithms
- Classifier-free guidance for conditional generation
- Trained on CIFAR-10 dataset
- FID reporting and ablation workflow

**Technologies:** PyTorch, CIFAR-10, UNet, attention mechanisms

**Use Cases:** Image generation, denoising, creative AI applications

---

### 2. Mini-GPT: Decoder-Only Transformer
**Directory:** `mini-gpt/`

A GPT-style language model implementing the complete transformer architecture from scratch.

**Key Features:**
- Multi-head self-attention with causal masking
- Pre-LayerNorm transformer blocks
- KV cache for efficient autoregressive generation
- Mixed precision training
- Multiple sampling strategies: greedy, top-k, top-p, temperature
- Configurations from small to medium-scale experiments

**Technologies:** PyTorch, Transformers, Attention, NLP

**Use Cases:** Text generation, language modeling, conversational AI

---

### 3. PPO from Scratch
**Directory:** `ppo-reinforcement-learning/`

A complete implementation of Proximal Policy Optimization for reinforcement learning.

**Key Features:**
- Clipped surrogate objective for stable learning
- Generalized Advantage Estimation
- Separate actor-critic networks
- Custom inventory management environment
- Trained on CartPole and custom supply-chain optimization
- Current inventory result uses random-policy baseline; classical EOQ / newsvendor / `(s,S)` baselines should be added before resume use

**Technologies:** PyTorch, Gymnasium, Reinforcement Learning, Policy Gradients

**Use Cases:** Operations research, inventory control, RL methodology

---

### 4. Mistral 7B LoRA / QLoRA Fine-Tuning
**Directory:** `mistral-7b-qlora-lora-finetuning/`

Instruction fine-tuning project currently under active training. It should be used on resumes only after training completes, evaluation is reproducible, and LoRA hyperparameter ablations are documented.

**Target Evidence Before Resume Use:**
- Base vs fine-tuned evaluation
- LoRA rank / alpha / dropout ablations
- QLoRA memory and throughput report
- Held-out instruction-following evaluation
- Clear limitations around dataset quality and hallucination risk

---

### 5. Credit Risk and Customer Lifecycle Analytics
**Directory:** `credit-risk-customer-lifecycle-analytics/`

End-to-end fintech data science project for credit risk, banking analytics, customer lifecycle, scorecards, and portfolio monitoring roles.

**Key Features:**
- Synthetic but realistic 25,000-customer banking portfolio with bureau, repayment, utilization, acquisition, churn, and margin signals
- Temporal validation by customer vintage month to avoid random-split leakage
- Model comparison across logistic scorecard, gradient boosting, and random forest
- ROC-AUC 0.763, KS 0.405, and 34.5% recall in the riskiest decile from the synthetic lifecycle pipeline
- Real OpenML German Credit benchmark: ROC-AUC 0.801 and KS 0.494 on 1,000 public records
- DuckDB SQL analytics layer for approval policy, realized default, churn, segment profitability, and acquisition-channel quality
- Permutation explainability, PSI drift monitoring, model card, executive brief, plots, and tests

**Technologies:** Python, pandas, scikit-learn, DuckDB, matplotlib, pytest

**Use Cases:** Credit risk, fintech analytics, data scientist, product analytics, banking DS, model governance

---

## Project Structure

```text
ML-PROJECTS/
- real-time-model-serving-monitoring/ # FastAPI model serving + monitoring
- diffusion-models/                   # DDPM image generation
- mini-gpt/                           # GPT-style language model
- ppo-reinforcement-learning/         # PPO RL implementation
- mistral-7b-qlora-lora-finetuning/   # Active LoRA / QLoRA fine-tuning work
- credit-risk-customer-lifecycle-analytics/ # Credit risk + customer lifecycle analytics
- README.md                           # This file
```

## Common Characteristics

All projects aim to demonstrate:
- **From First Principles:** Core algorithms implemented explicitly where educational value matters
- **Production Quality:** Modular design, documentation, error handling, and tests where appropriate
- **Research Grade:** Clear baselines, reproducible metrics, ablations, and honest limitations
- **Interview Readiness:** Each project should have defensible bullets, caveats, and walkthroughs

## Technologies Used

- **Deep Learning:** PyTorch, Transformers, diffusion models, reinforcement learning
- **MLOps:** FastAPI, Docker, Pydantic, pytest, model monitoring
- **Python Stack:** NumPy, pandas, scikit-learn, matplotlib, TensorBoard where relevant
- **Environments:** CPU-friendly tests by default; GPU required only for heavier training projects

## Getting Started

Each project has its own README with detailed instructions.

```bash
# Real-time model serving
cd real-time-model-serving-monitoring/
pip install -r requirements.txt
python scripts/train_model.py
python -m pytest -q
uvicorn src.ml_monitoring.app:app --host 0.0.0.0 --port 8000

# Diffusion Models
cd diffusion-models/
pip install -r requirements.txt
python training/train_model.py --config configs/base.yaml --experiment baseline

# Mini-GPT
cd mini-gpt/
pip install -r requirements.txt
python training/train_model.py --config configs/small.yaml

# PPO
cd ppo-reinforcement-learning/
pip install -r requirements.txt
python training/train_agent.py --env CartPole-v1 --total-timesteps 50000
```

## Resume Use

For AI/ML Engineer resumes, prioritize:
1. Real-Time Model Serving and Monitoring Pipeline
2. Mini-GPT
3. DDPM / diffusion project
4. Mistral LoRA / QLoRA only after training and ablation results are complete

FlowFinance production AI work should be listed as venture / founder experience, not as a project.

For Data Scientist / fintech analytics resumes, prioritize:
1. Credit Risk and Customer Lifecycle Analytics
2. SQL Operations Analytics
3. Demand Forecasting + Safety Stock
4. A/B Testing and Funnel Analytics

## License

Each project is individually licensed. See the respective license files in each project directory.
