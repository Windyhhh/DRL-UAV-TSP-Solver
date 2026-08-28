<div align="center">

# 🚁 DRL-UAV-TSP-Solver

### Deep RL for the UAV travelling-salesman problem.

Attention-based A2C solves the drone TSP (TSPD) at multiple scales — with rich visualization.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

**DRL-UAV-TSP-Solver** solves the **UAV travelling-salesman problem (TSPD)** — jointly optimizing drone and ground-vehicle routes — with a **deep-reinforcement-learning** approach built on **attention mechanisms and A2C**. It supports multiple scales (n = 11, 15, 20, 50, 100) with rich visualization.

> [!NOTE]
> 中文项目：基于深度强化学习（注意力机制 + A2C）的无人机旅行商问题（TSPD）求解——支持 n=11~100 多规模路径优化。

---

## Features

- **DRL solver** — attention-based A2C for combinatorial optimization.
- **Multi-scale** — n = 11 / 15 / 20 / 50 / 100.
- **Performance** — on n=100, ~40% faster and ~15% better solutions than traditional heuristics.
- **Visualization** — progress & result visualizers.
- **Applications** — logistics, grid inspection, rescue routing.

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/DRL-UAV-TSP-Solver.git
cd DRL-UAV-TSP-Solver

pip install -r requirements.txt

python src/main.py          # train / solve
python src/visualize.py     # visualize routes
```

---

## Project Structure

```
DRL-UAV-TSP-Solver/
├── src/                    # model, A2C training, solver
├── viz/                    # route visualization
├── models/                 # pretrained weights
└── docs/                   # usage, blog
```

---

## License

MIT — free to use, modify and distribute.
