#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式TSP Drone可视化工具
Interactive TSP Drone Visualization Tool

功能特点：
- 交互式参数调整
- 实时可视化
- 多实例对比
- 训练进度监控
- 解决方案导出
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, RadioButtons
import os
import sys
import torch
import argparse
import json
from datetime import datetime
import seaborn as sns

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.utils.env_no_comb import Env, DataGenerator
from src.utils.options import ParseParams
from src.model.nnets import Actor, Critic
from src.utils.agent import A2CAgent

class InteractiveTSPVisualizer:
    """交互式TSP Drone可视化器"""
    
    def __init__(self, n_nodes=11, use_pretrained=True):
        self.n_nodes = n_nodes
        self.use_pretrained = use_pretrained
        self.config = self.get_default_config()
        self.problem_data = None
        self.solution = None
        self.fig = None
        self.axes = {}
        self.sliders = {}
        self.buttons = {}
        
        # 加载数据和模型
        self.load_data_and_models()
        
    def get_default_config(self):
        """获取默认配置"""
        return {
            'n_nodes': self.n_nodes,
            'batch_size': 1,
            'test_size': 1,
            'random_seed': 42,
            'data_dir': 'data',
            'save_path': 'trained_models/',
            'log_dir': 'logs',
            'hidden_dim': 256,
            'decode_len': 30,
            'v_t': 1,
            'v_d': 2,
            'R': 150,
            'max_w': 2.5,
            'stdout_print': False
        }
    
    def load_data_and_models(self):
        """加载数据和预训练模型"""
        print(f"Loading data and models for {self.n_nodes} nodes...")
        
        # 加载测试数据
        dataGen = DataGenerator(self.config)
        test_data = dataGen.get_test_all()
        self.problem_data = test_data[0]  # 取第一个实例
        
        # 加载预训练模型
        if self.use_pretrained:
            self.actor, self.critic = self.load_pretrained_models()
        else:
            self.actor, self.critic = None, None
    
    def load_pretrained_models(self):
        """加载预训练模型"""
        try:
            actor = Actor(self.config['hidden_dim'])
            critic = Critic(self.config['hidden_dim'])
            
            save_path = self.config['save_path']
            actor_path = os.path.join(save_path, f'n{self.n_nodes}', 'best_model_actor_truck_params.pkl')
            critic_path = os.path.join(save_path, f'n{self.n_nodes}', 'best_model_critic_params.pkl')
            
            if os.path.exists(actor_path):
                actor.load_state_dict(torch.load(actor_path, map_location='cpu'))
                print(f"Loaded actor model from {actor_path}")
            
            if os.path.exists(critic_path):
                critic.load_state_dict(torch.load(critic_path, map_location='cpu'))
                print(f"Loaded critic model from {critic_path}")
            
            return actor, critic
        except Exception as e:
            print(f"Warning: Could not load pretrained models: {e}")
            return None, None
    
    def solve_current_problem(self):
        """解决当前问题"""
        if self.actor is None:
            print("No pretrained model available")
            return
        
        try:
            # 创建环境和智能体
            env = Env(self.config, self.problem_data.reshape(1, -1, 3))
            dataGen = DataGenerator(self.config)
            agent = A2CAgent(self.actor, self.critic, self.config, env, dataGen)
            
            # 求解
            with torch.no_grad():
                solution_data = agent.test()
                self.solution = {
                    'truck_route': solution_data.get('truck_route', []),
                    'total_time': solution_data.get('total_time', 0),
                    'reward': solution_data.get('reward', 0)
                }
            
            print(f"Solution found: Total time = {self.solution['total_time']:.2f}")
            
        except Exception as e:
            print(f"Error solving problem: {e}")
            self.solution = None
    
    def create_main_interface(self):
        """创建主界面"""
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('TSP Drone 交互式可视化工具', fontsize=16, fontweight='bold')
        
        # 创建网格布局
        gs = self.fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # 主可视化区域
        self.axes['main'] = self.fig.add_subplot(gs[0, :2])
        
        # 问题信息区域
        self.axes['problem'] = self.fig.add_subplot(gs[0, 2])
        
        # 解决方案信息区域
        self.axes['solution'] = self.fig.add_subplot(gs[0, 3])
        
        # 统计图表区域
        self.axes['stats'] = self.fig.add_subplot(gs[1, :])
        
        # 控制面板区域
        self.axes['controls'] = self.fig.add_subplot(gs[2, :])
        self.axes['controls'].axis('off')
        
        # 创建控制组件
        self.create_controls()
        
        # 初始化可视化
        self.update_visualization()
    
    def create_controls(self):
        """创建控制组件"""
        # 问题选择滑块
        ax_prob_slider = plt.axes([0.1, 0.02, 0.3, 0.03])
        self.sliders['problem_idx'] = Slider(ax_prob_slider, '问题编号', 0, 99, valinit=0, valstep=1)
        self.sliders['problem_idx'].on_changed(self.on_problem_change)
        
        # 节点数量滑块
        ax_nodes_slider = plt.axes([0.45, 0.02, 0.3, 0.03])
        self.sliders['n_nodes'] = Slider(ax_nodes_slider, '节点数量', 11, 100, valinit=self.n_nodes, valstep=1)
        self.sliders['n_nodes'].on_changed(self.on_nodes_change)
        
        # 求解按钮
        ax_solve_btn = plt.axes([0.8, 0.02, 0.08, 0.04])
        self.buttons['solve'] = Button(ax_solve_btn, '求解')
        self.buttons['solve'].on_clicked(self.on_solve_click)
        
        # 导出按钮
        ax_export_btn = plt.axes([0.9, 0.02, 0.08, 0.04])
        self.buttons['export'] = Button(ax_export_btn, '导出')
        self.buttons['export'].on_clicked(self.on_export_click)
    
    def on_problem_change(self, val):
        """问题索引改变回调"""
        problem_idx = int(val)
        print(f"Loading problem {problem_idx}...")
        # 这里可以加载不同的测试实例
        self.update_visualization()
    
    def on_nodes_change(self, val):
        """节点数量改变回调"""
        new_n_nodes = int(val)
        if new_n_nodes != self.n_nodes:
            print(f"Changing node count to {new_n_nodes}...")
            self.n_nodes = new_n_nodes
            self.config['n_nodes'] = new_n_nodes
            self.load_data_and_models()
            self.update_visualization()
    
    def on_solve_click(self, event):
        """求解按钮点击回调"""
        print("Solving current problem...")
        self.solve_current_problem()
        self.update_visualization()
    
    def on_export_click(self, event):
        """导出按钮点击回调"""
        if self.solution:
            self.export_solution()
        else:
            print("No solution to export")
    
    def update_visualization(self):
        """更新可视化"""
        self.clear_axes()
        self.plot_problem()
        self.plot_solution()
        self.update_info_panels()
        self.update_stats()
        plt.draw()
    
    def clear_axes(self):
        """清空所有绘图区域"""
        for ax in self.axes.values():
            if ax.name != 'controls':
                ax.clear()
    
    def plot_problem(self):
        """绘制问题实例"""
        ax = self.axes['main']
        
        coordinates = self.problem_data[:, :2]
        demands = self.problem_data[:, 2]
        n_nodes = len(coordinates)
        depot_idx = n_nodes - 1
        
        # 绘制节点
        customer_indices = np.where(demands > 0)[0]
        
        ax.scatter(coordinates[customer_indices, 0], coordinates[customer_indices, 1], 
                  c='red', s=100, alpha=0.7, label='客户', zorder=3)
        
        ax.scatter(coordinates[depot_idx, 0], coordinates[depot_idx, 1], 
                  c='blue', s=200, marker='s', label='仓库', zorder=3)
        
        # 添加节点标签
        for i, (x, y) in enumerate(coordinates):
            if demands[i] > 0:
                ax.annotate(f'{i}', (x, y), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8, fontweight='bold')
            else:
                ax.annotate(f'仓库', (x, y), xytext=(5, 5), 
                          textcoords='offset points', fontsize=10, fontweight='bold', color='blue')
        
        ax.set_title(f'TSP Drone 问题实例 (节点数: {n_nodes})', fontsize=12, fontweight='bold')
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置相等的坐标轴比例
        ax.set_aspect('equal')
    
    def plot_solution(self):
        """绘制解决方案"""
        ax = self.axes['main']
        
        if self.solution is None:
            ax.text(0.5, 0.5, '点击"求解"按钮获取解决方案', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
            return
        
        coordinates = self.problem_data[:, :2]
        truck_route = self.solution.get('truck_route', [])
        
        # 绘制卡车路径
        if truck_route and len(truck_route) > 1:
            truck_coords = coordinates[truck_route]
            ax.plot(truck_coords[:, 0], truck_coords[:, 1], 
                   'b-', linewidth=3, label='卡车路径', alpha=0.8, zorder=1)
            
            # 添加方向箭头
            for i in range(len(truck_route) - 1):
                start = coordinates[truck_route[i]]
                end = coordinates[truck_route[i + 1]]
                dx, dy = end[0] - start[0], end[1] - start[1]
                mid_point = (start + end) / 2
                ax.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.8))
        
        # 高亮显示路径节点
        if truck_route:
            route_coords = coordinates[truck_route]
            ax.scatter(route_coords[:, 0], route_coords[:, 1], 
                      c='green', s=150, alpha=0.9, label='路径节点', zorder=4)
    
    def update_info_panels(self):
        """更新信息面板"""
        # 问题信息
        problem_ax = self.axes['problem']
        problem_ax.axis('off')
        
        coordinates = self.problem_data[:, :2]
        demands = self.problem_data[:, 2]
        n_nodes = len(coordinates)
        depot_idx = n_nodes - 1
        customer_indices = np.where(demands > 0)[0]
        
        problem_info = f"""问题信息:
        
节点总数: {n_nodes}
客户数量: {len(customer_indices)}
仓库编号: {depot_idx}

总需求: {np.sum(demands):.1f}
平均需求: {np.mean(demands[demands > 0]):.2f}
最大需求: {np.max(demands):.1f}

卡车速度: {self.config['v_t']} 单位/时间
无人机速度: {self.config['v_d']} 单位/时间
电池续航: {self.config['R']} 时间单位
        """
        
        problem_ax.text(0.05, 0.95, problem_info, transform=problem_ax.transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # 解决方案信息
        solution_ax = self.axes['solution']
        solution_ax.axis('off')
        
        if self.solution:
            total_time = self.solution.get('total_time', 0)
            truck_visits = len(set(self.solution.get('truck_route', [])))
            
            solution_info = f"""解决方案:
            
总时间: {total_time:.2f}
卡车访问节点数: {truck_visits}
路径长度: {len(self.solution.get('truck_route', []))}

效率指标:
访问率: {truck_visits/n_nodes*100:.1f}%
效率比: {n_nodes/truck_visits:.2f}
        """
        else:
            solution_info = """解决方案:
            
尚未求解
点击"求解"按钮
获取最优路径
        """
        
        solution_ax.text(0.05, 0.95, solution_info, transform=solution_ax.transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    def update_stats(self):
        """更新统计图表"""
        stats_ax = self.axes['stats']
        stats_ax.clear()
        
        # 这里可以添加各种统计图表
        # 例如：节点分布、距离分析、需求分布等
        
        coordinates = self.problem_data[:, :2]
        demands = self.problem_data[:, 2]
        
        # 创建子图
        gs = stats_ax.get_gridspec()
        stats_ax.remove()
        
        # 创建2x2的子图
        sub_ax1 = self.fig.add_subplot(gs[1, 0])
        sub_ax2 = self.fig.add_subplot(gs[1, 1])
        sub_ax3 = self.fig.add_subplot(gs[1, 2])
        sub_ax4 = self.fig.add_subplot(gs[1, 3])
        
        # 节点分布直方图
        sub_ax1.hist(coordinates[:, 0], bins=10, alpha=0.7, color='skyblue')
        sub_ax1.set_title('X坐标分布')
        sub_ax1.set_xlabel('X坐标')
        sub_ax1.set_ylabel('频次')
        
        sub_ax2.hist(coordinates[:, 1], bins=10, alpha=0.7, color='lightcoral')
        sub_ax2.set_title('Y坐标分布')
        sub_ax2.set_xlabel('Y坐标')
        sub_ax2.set_ylabel('频次')
        
        # 需求分布
        customer_demands = demands[demands > 0]
        sub_ax3.bar(range(len(customer_demands)), customer_demands, alpha=0.7, color='gold')
        sub_ax3.set_title('客户需求分布')
        sub_ax3.set_xlabel('客户编号')
        sub_ax3.set_ylabel('需求')
        
        # 距离矩阵热力图（简化版）
        n_nodes = len(coordinates)
        distances = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                distances[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
        
        im = sub_ax4.imshow(distances, cmap='viridis', aspect='auto')
        sub_ax4.set_title('距离矩阵')
        sub_ax4.set_xlabel('节点')
        sub_ax4.set_ylabel('节点')
        plt.colorbar(im, ax=sub_ax4, shrink=0.8)
        
        self.fig.tight_layout()
    
    def export_solution(self):
        """导出解决方案"""
        if not self.solution:
            print("No solution to export")
            return
        
        # 准备导出数据
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'n_nodes': self.n_nodes,
            'problem_data': self.problem_data.tolist(),
            'solution': self.solution,
            'config': self.config
        }
        
        # 保存为JSON文件
        filename = f"tsp_solution_{self.n_nodes}_nodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Solution exported to {filename}")
        
        # 保存为图像
        if self.fig:
            img_filename = filename.replace('.json', '.png')
            self.fig.savefig(img_filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {img_filename}")
    
    def run(self):
        """运行可视化器"""
        self.create_main_interface()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Interactive TSP Drone Visualizer')
    parser.add_argument('--n_nodes', type=int, default=11, help='Number of nodes')
    parser.add_argument('--no_pretrained', action='store_true', help='Do not use pretrained models')
    
    args = parser.parse_args()
    
    print("启动交互式TSP Drone可视化工具...")
    print("=" * 50)
    print("功能说明:")
    print("1. 使用滑块调整问题参数")
    print("2. 点击'求解'按钮获取最优路径")
    print("3. 点击'导出'按钮保存结果")
    print("4. 关闭窗口退出程序")
    print("=" * 50)
    
    try:
        visualizer = InteractiveTSPVisualizer(
            n_nodes=args.n_nodes,
            use_pretrained=not args.no_pretrained
        )
        visualizer.run()
    except Exception as e:
        print(f"Error running visualizer: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()