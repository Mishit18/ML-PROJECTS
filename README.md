# Machine Learning Projects Portfolio

A collection of production-grade machine learning implementations built from first principles. Each project demonstrates deep understanding of core algorithms through clean, modular code and rigorous validation.

## Projects Overview

### 1. Diffusion Models from Scratch
**Directory:** `diffusion-models/`

A complete implementation of Denoising Diffusion Probabilistic Models (DDPM) for image generation.

**Key Features:**
- Forward and reverse diffusion processes with multiple noise schedules
- Custom UNet architecture with attention mechanisms
- DDPM and DDIM sampling algorithms
- Classifier-free guidance for conditional generation
- Trained on CIFAR-10 dataset (500 epochs)
- FID Score: 10-15, Model Size: 35M parameters

**Technologies:** PyTorch, CIFAR-10, UNet, Attention Mechanisms

**Use Cases:** Image generation, denoising, creative AI applications

---

### 2. Mini-GPT: Decoder-Only Transformer
**Directory:** `mini-gpt/`

A GPT-style language model implementing the complete transformer architecture from scratch.

**Key Features:**
- Multi-head self-attention with causal masking
- Pre-LayerNorm transformer blocks
- KV cache for efficient autoregressive generation (10-15x speedup)
- Mixed precision training (FP16/FP32)
- Multiple sampling strategies (greedy, top-k, top-p, temperature)
- Configurations: Small (25M), Base (117M), Medium (350M) parameters

**Technologies:** PyTorch, Transformers, Attention, NLP

**Use Cases:** Text generation, language modeling, conversational AI

---

### 3. PPO from Scratch
**Directory:** `ppo-reinforcement-learning/`

A complete implementation of Proximal Policy Optimization for reinforcement learning.

**Key Features:**
- Clipped surrogate objective for stable learning
- Generalized Advantage Estimation (GAE)
- Separate actor-critic networks
- Custom inventory management environment
- Trained on CartPole and custom supply chain optimization
- 71% improvement over random policy on inventory management

**Technologies:** PyTorch, Gymnasium, Reinforcement Learning, Policy Gradients

**Use Cases:** Game AI, robotics control, operations research, RLHF

---

## Project Structure

```
ML-PROJECTS/
├── diffusion-models/           # DDPM image generation
├── mini-gpt/                   # GPT-style language model
├── ppo-reinforcement-learning/ # PPO RL implementation
└── README.md                   # This file
```

## Common Characteristics

All projects demonstrate:
- **From First Principles:** No high-level abstractions, every component implemented explicitly
- **Production Quality:** Modular design, comprehensive documentation, proper error handling
- **Research Grade:** Mathematical rigor, proper citations, reproducible results
- **Educational Value:** Clear code structure, detailed comments, suitable for learning

## Technologies Used

- **Deep Learning Framework:** PyTorch 2.0+
- **Python:** 3.8+
- **Key Libraries:** NumPy, Matplotlib, TensorBoard
- **Environments:** CUDA 11.8+ (optional, for GPU acceleration)

## Getting Started

Each project has its own README with detailed instructions. Navigate to the respective directory and follow the setup instructions:

```bash
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

## Project Highlights

### Diffusion Models
- Implements state-of-the-art image generation
- Achieves competitive FID scores on CIFAR-10
- Demonstrates understanding of probabilistic modeling

### Mini-GPT
- Complete transformer implementation without abstractions
- Efficient inference with KV caching
- Demonstrates understanding of attention mechanisms and language modeling

### PPO
- Stable reinforcement learning with clipped objective
- Custom environment design (inventory management)
- Demonstrates understanding of policy gradient methods

## License

Each project is individually licensed. See the respective `license.txt` files in each project directory.

Copyright (c) 2026. All Rights Reserved.

---

**Note:** These implementations prioritize clarity and correctness over performance optimization, making them suitable for educational purposes, research, and as foundations for production systems.
