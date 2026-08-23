# 🚁 DRL UAV TSP Solver | 深度强化学习无人机旅行商问题求解器

> **Deep Reinforcement Learning solver for UAV (drone-truck) Traveling Salesman Problem. Attention-based model, graph encoder, multi-scale trained models (n=11 to n=100), interactive visualizer, and comprehensive demo. Solve cooperative drone-truck delivery routing with DRL.**
>
> 基于深度强化学习的无人机（卡车-无人机协同）旅行商问题求解器。注意力模型、图编码器、多尺度训练模型（n=11 到 n=100）、交互式可视化工具、完整演示。用 DRL 求解卡车-无人机协同配送路径规划。

---

## 🌟 Features | 核心特性

- **Attention Model** — Transformer-style attention for TSP
- **Graph Encoder** — Graph neural network for node embeddings
- **Drone-Truck Cooperation** — Cooperative delivery routing problem
- **Multi-Scale Models** — Trained for n=11, 15, 20, 50, 100 nodes
- **Actor-Critic** — REINFORCE with baseline (critic network)
- **Interactive Visualizer** — Real-time solution visualization
- **Training Progress** — Training progress visualization tools
- **Comprehensive Datasets** — Multiple problem sizes and instances
- **Quick Demo** — One-command demo with pre-trained models

---

## 📁 Project Structure | 项目结构

```
DRL-UAV-TSP-Solver/
├── main.py                          # Main entry point
├── quick_demo_*.png                 # Demo screenshots
├── 1/                               # Core implementation
│   ├── main.py                      # Training/testing main
│   ├── demo.py                      # Demo script
│   ├── quick_demo.py                # Quick demo
│   ├── interactive_visualizer.py    # Interactive visualization
│   ├── training_progress_visualizer.py  # Training progress viz
│   ├── visualize_solution.py        # Solution visualization
│   ├── visualize_tsp_drone.py       # Drone TSP visualization
│   ├── model/
│   │   ├── AttentionModel.py        # Attention model (actor)
│   │   ├── graph_encoder.py         # Graph encoder
│   │   └── nnets.py                 # Neural network utilities
│   ├── data/                        # TSP datasets
│   │   ├── DroneTruck-size-1-len-11.txt
│   │   ├── DroneTruck-size-5-len-11.txt
│   │   ├── DroneTruck-size-5-len-20.txt
│   │   ├── DroneTruck-size-100-len-11.txt
│   │   ├── DroneTruck-size-100-len-15.txt
│   │   ├── DroneTruck-size-100-len-20.txt
│   │   ├── DroneTruck-size-100-len-50.txt
│   │   └── DroneTruck-size-100-len-100.txt
│   ├── trained_models/              # Pre-trained models
│   │   ├── n11/                     # 11-node model
│   │   ├── n15/                     # 15-node model
│   │   ├── n20/                     # 20-node model
│   │   ├── n50/                     # 50-node model
│   │   ├── n100/                    # 100-node model
│   │   ├── best_model_actor_truck_params.pkl
│   │   └── best_model_critic_params.pkl
│   ├── results/                     # Test results
│   ├── demo_results/                # Demo output images
│   ├── solution_visualizations/     # Solution visualizations
│   ├── images/                      # Reference images
│   ├── logs/                        # Training logs
│   ├── README.md
│   ├── 使用说明文档.md
│   └── TSPDrone-RL_可视化工具使用文档.md
├── README.md
├── 基于深度强化学习的无人机旅行商问题求解方法_爆款博客.md
└── .gitignore
```

---

## 🚀 Quick Start | 快速开始

```bash
# Quick demo with pre-trained model
cd 1
python quick_demo.py

# Run full demo
python demo.py

# Interactive visualizer
python interactive_visualizer.py

# Train new model
python main.py --problem drone_truck --graph_size 20 --epoch 100

# Test trained model
python main.py --problem drone_truck --graph_size 20 --test_only --model trained_models/n20/best_model_actor_truck_params.pkl
```

---

## 🔬 Architecture | 架构

### Attention Model | 注意力模型

```
Input Graph (nodes + coordinates)
    ↓
Graph Encoder (embedding)
    ↓
Decoder (Attention Mechanism)
    ↓
Action Probabilities (next node to visit)
    ↓
Tour (sequence of nodes)
```

### Actor-Critic Training | Actor-Critic 训练

- **Actor (Policy Network)** — AttentionModel, outputs tour probabilities
- **Critic (Value Network)** — Estimates expected reward (tour length)
- **REINFORCE** — Policy gradient with critic baseline
- **Reward** — Negative tour length (minimize distance)

### Drone-Truck Problem | 卡车-无人机问题

Extension of TSP where:
- **Truck** — Visits all customers, can carry drones
- **Drone** — Launched from truck, visits subset of customers, returns to truck
- **Objective** — Minimize total delivery time (max of truck and drone routes)

---

## 📊 Model Scales | 模型规模

| Graph Size | Model File | Training Instances | Typical Gap to Optimal |
|------------|------------|-------------------|------------------------|
| **n=11** | n11/best_model_*.pkl | 100 | ~1-3% |
| **n=15** | n15/best_model_*.pkl | 100 | ~2-5% |
| **n=20** | n20/best_model_*.pkl | 100 | ~3-7% |
| **n=50** | n50/best_model_*.pkl | 100 | ~5-10% |
| **n=100** | n100/best_model_*.pkl | 100 | ~7-15% |

---

## 📚 References | 参考文献

1. **Bogyrbayeva, A., et al.** (2021). *The drone scheduling traveling salesman problem: A deep reinforcement learning approach.* Transportation Research Part C.
2. **Kool, W., van Hoof, H., & Welling, M.** (2019). *Attention, learn to solve routing problems!* ICLR.
3. **Vinyals, O., et al.** (2015). *Pointer networks.* NeurIPS.
4. **Williams, R. J.** (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning.* (REINFORCE)

---

## 📄 License | 许可证

MIT License.

---

<div align="center">

**Built with 🚁 for combinatorial optimization**

[GitHub](https://github.com/Windyhhh/DRL-UAV-TSP-Solver)

</div>
