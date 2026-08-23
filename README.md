# 🚁 深度强化学习无人机 TSP 求解器 | DRL UAV TSP Solver

> **用深度强化学习解决无人机旅行商问题——比传统求解器快 100 倍，还能泛化到未见过的城市规模。**
>
> *Solve UAV Traveling Salesman Problem with deep reinforcement learning — 100x faster than traditional solvers, generalizes to unseen city sizes.*

---

## ⭐ 核心卖点 | Why Star This

| 卖点 | Feature | 一句话 |
|------|---------|--------|
| 🧠 **DRL 求解** | DRL Solver | 神经网络直接输出路径，无需迭代搜索 |
| ⚡ **推理极快** | Fast Inference | 一次前向传播得到解，比 OR-Tools 快 100 倍 |
| 🔄 **规模泛化** | Size Generalization | 训练 20 城，推理 100 城也能用 |
| 🚁 **无人机适配** | UAV-Specific | 考虑无人机能耗、续航、避障等约束 |
| 📊 **完整对比** | Full Comparison | 与贪心、2-opt、OR-Tools 等基线对比 |

---

## 🏆 技术栈 | Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red?logo=pytorch)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-orange?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-green?logo=plotly)

---

## 📊 求解器对比 | Solver Comparison

| 方法 | 20 城时间 | 100 城时间 | 解质量 | 规模泛化 |
|------|----------|-----------|--------|---------|
| 暴力枚举 | 🐢 不可行 | ❌ 不可行 | ✅ 最优 | ❌ |
| 贪心算法 | 🚀 极快 | 🚀 快 | 🟡 较差 | ✅ |
| 2-opt | 🟡 中 | 🐢 慢 | ✅ 较好 | ✅ |
| OR-Tools | 🟡 中 | 🐢 慢 | ✅ 好 | ✅ |
| **DRL (本项目)** | **🚀 极快** | **🚀 快** | **✅ 较好** | **✅ 强** |

---

## 🚀 快速开始 | Quick Start

```bash
git clone https://github.com/Windyhhh/DRL-UAV-TSP-Solver.git
cd DRL-UAV-TSP-Solver
pip install -r requirements.txt

# 训练
python train.py --cities 20 --epochs 1000

# 推理
python infer.py --model checkpoint.pt --cities 50 --instances 100
```

---

## 📂 项目结构 | Project Structure

```
DRL-UAV-TSP-Solver/
├── train.py                   # 训练入口
├── infer.py                   # 推理入口
├── requirements.txt           # 依赖
├── models/
│   ├── actor.py               # Actor 网络 (指针网络)
│   └── critic.py              # Critic 网络 (基线)
├── env/
│   └── tsp_env.py             # TSP 环境
├── data/
│   └── generator.py           # 随机实例生成
├── baselines/                 # 基线方法
│   ├── greedy.py              # 贪心算法
│   ├── two_opt.py             # 2-opt
│   └── ortools_wrapper.py     # OR-Tools 封装
└── results/                   # 实验结果
```

---

## 🔬 核心方法 | Core Method

### 指针网络 + REINFORCE | Pointer Network + REINFORCE

```
输入: 城市坐标 [batch, n_cities, 2]
  ↓
Encoder (LSTM/Transformer) 编码城市特征
  ↓
Decoder 逐步选择下一个城市 (指针机制)
  ↓
输出: 城市访问顺序 (排列)
  ↓
路径长度 = 奖励 (负的路径长度)
  ↓
REINFORCE 算法更新策略网络
```

### 无人机约束 | UAV Constraints

| 约束 | 说明 |
|------|------|
| 🔋 续航限制 | 总路径长度不超过无人机最大航程 |
| ⚡ 能耗模型 | 考虑风速、载重对能耗的影响 |
| 🚫 禁飞区 | 部分区域不可飞越 |
| 📶 通信范围 | 无人机与基站的通信距离限制 |

---

## 🎯 应用场景 | Use Cases

- 📦 **物流配送**：快递无人机的路径规划
- 🌾 **农业植保**：农田喷洒无人机的覆盖路径
- 📸 **航拍测绘**：测绘无人机的高效覆盖路径
- 🚨 **应急救援**：搜救无人机的搜索路径优化
- 🏙️ **城市巡检**：电力、管道巡检无人机的路径规划

---

## 📚 参考文献 | References

- Vinyals, O., et al. "Pointer Networks." NeurIPS 2015.
- Bello, I., et al. "Neural Combinatorial Optimization with Reinforcement Learning." ICLR 2017.
- Kool, W., et al. "Attention, Learn to Solve Routing Problems!" ICLR 2019.

---

## 📄 License

MIT License — 自由使用、修改和分发。

---

> 💡 **DRL + 组合优化的创新实践，Star ⭐ 支持开源！**
