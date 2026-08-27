<div align="center">

# 🚁 DRL-UAV-TSP-Solver

### Deep reinforcement learning for UAV + truck TSP routing.

Attention model with a graph encoder, multi-scale (n = 11–100) and interactive visualization.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

**DRL-UAV-TSP-Solver** solves the **UAV + truck TSP** routing problem with deep reinforcement learning — an **attention model** with a graph encoder, trained across scales (**n = 11–100**) and bundled with interactive visualization.

> [!NOTE]
> 中文项目：深度强化学习无人机路径——UAV 无人机 + 卡车 TSP 求解，Attention 模型，图编码器，多尺度模型（n=11-100），交互式可视化。

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/DRL-UAV-TSP-Solver.git
cd DRL-UAV-TSP-Solver

# Interactive demo / visualization
python 1/demo.py
```

Instance data (DroneTruck-size-{k}-len-{n}) ships in `1/data/`, with sample results in `1/demo_results/`.

---

## Features

- **Attention + graph encoder** — DRL model for routing.
- **Multi-scale** — trained for n = 11 ~ 100.
- **Interactive visualization** — route rendering for solutions.

---

## Project Structure

```
DRL-UAV-TSP-Solver/
├── 1/
│   ├── demo.py                  # interactive demo
│   ├── data/                    # DroneTruck instances (size / len)
│   ├── demo_results/            # solution PNGs
│   └── images/                  # optimal-solution figures
└── README.md
```

---

## License

MIT — free to use, modify and distribute.
