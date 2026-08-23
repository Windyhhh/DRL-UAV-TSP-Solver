#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速演示脚本 - 快速查看TSP Drone的图形化效果
Quick Demo Script - Quick TSP Drone Visualization

这个脚本提供了最简单的图形化演示，无需复杂配置
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os
import torch
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from utils.env_no_comb import Env, DataGenerator
from utils.options import ParseParams
from model.nnets import Actor, Critic
from utils.agent import A2CAgent

def create_sample_data(n_nodes=11):
    """创建示例数据"""
    np.random.seed(42)  # 固定随机种子以获得可重现的结果
    
    # 生成随机坐标
    coordinates = np.random.uniform(10, 90, size=(n_nodes-1, 2))
    
    # 添加仓库（最后一个节点）
    depot = np.array([[50, 50]])  # 仓库在中心位置
    coordinates = np.vstack([coordinates, depot])
    
    # 设置需求（客户需求为1.0，仓库需求为0.0）
    demands = np.ones(n_nodes)
    demands[-1] = 0.0  # 仓库无需求
    
    # 组合数据 [x, y, demand]
    data = np.column_stack([coordinates, demands])
    
    return data

def load_pretrained_model(n_nodes=11):
    """加载预训练模型"""
    try:
        config = {
            'hidden_dim': 256,
            'n_nodes': n_nodes,
            'save_path': 'trained_models/',
            'data_dir': 'data',
            'batch_size': 1,
            'test_size': 1,
            'random_seed': 42,
            'decode_len': 30,
            'v_t': 1,
            'v_d': 2,
            'R': 150,
            'stdout_print': False
        }
        
        actor = Actor(config['hidden_dim'])
        critic = Critic(config['hidden_dim'])
        
        save_path = config['save_path']
        actor_path = os.path.join(save_path, f'n{n_nodes}', 'best_model_actor_truck_params.pkl')
        critic_path = os.path.join(save_path, f'n{n_nodes}', 'best_model_critic_params.pkl')
        
        if os.path.exists(actor_path):
            actor.load_state_dict(torch.load(actor_path, map_location='cpu'))
            print(f"✓ 已加载预训练模型: {actor_path}")
        else:
            print(f"⚠ 未找到预训练模型: {actor_path}")
            return None, None, config
        
        if os.path.exists(critic_path):
            critic.load_state_dict(torch.load(critic_path, map_location='cpu'))
            print(f"✓ 已加载预训练模型: {critic_path}")
        
        return actor, critic, config
        
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        return None, None, None

def solve_problem_with_model(data, actor, critic, config):
    """使用训练好的模型解决问题"""
    if actor is None or critic is None:
        print("没有可用的预训练模型，跳过求解")
        return None
    
    try:
        # 创建环境和智能体
        env = Env(config, data.reshape(1, -1, 3))
        dataGen = type('DataGen', (), {'get_test_all': lambda self: data.reshape(1, -1, 3)})()
        agent = A2CAgent(actor, critic, config, env, dataGen)
        
        # 求解
        with torch.no_grad():
            result = agent.test()
            
        solution = {
            'truck_route': result.get('truck_route', []),
            'total_time': result.get('total_time', 0),
            'reward': result.get('reward', 0)
        }
        
        print(f"✓ 求解完成: 总时间 = {solution['total_time']:.2f}")
        return solution
        
    except Exception as e:
        print(f"✗ 求解失败: {e}")
        return None

def create_greedy_solution(data):
    """创建简单的贪心解决方案作为演示"""
    coordinates = data[:, :2]
    demands = data[:, 2]
    n_nodes = len(coordinates)
    depot_idx = n_nodes - 1
    
    # 简单的最近邻贪心算法
    route = [depot_idx]  # 从仓库开始
    remaining = set(range(n_nodes - 1))  # 除去仓库的所有节点
    
    current = depot_idx
    while remaining:
        # 找到距离最近的未访问节点
        distances = [np.linalg.norm(coordinates[current] - coordinates[i]) for i in remaining]
        nearest = min(remaining, key=lambda i: distances[list(remaining).index(i)])
        route.append(nearest)
        remaining.remove(nearest)
        current = nearest
    
    route.append(depot_idx)  # 回到仓库
    
    # 计算总距离
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += np.linalg.norm(coordinates[route[i]] - coordinates[route[i+1]])
    
    return {
        'truck_route': route,
        'total_time': total_distance,  # 简化的时间计算
        'reward': -total_distance  # 负的总距离作为奖励
    }

def visualize_tsp_drone_comprehensive(data, solution=None, save_path=None):
    """综合可视化TSP Drone问题"""
    
    coordinates = data[:, :2]
    demands = data[:, 2]
    n_nodes = len(coordinates)
    depot_idx = n_nodes - 1
    customer_indices = np.where(demands > 0)[0]
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TSP Drone 快速演示 - 图形化可视化', fontsize=16, fontweight='bold')
    
    # === 子图1: 问题设置 ===
    ax1.set_title('问题设置', fontsize=12, fontweight='bold')
    
    # 绘制节点
    ax1.scatter(coordinates[customer_indices, 0], coordinates[customer_indices, 1], 
               c='red', s=120, alpha=0.8, label=f'客户 ({len(customer_indices)}个)', zorder=3)
    ax1.scatter(coordinates[depot_idx, 0], coordinates[depot_idx, 1], 
               c='blue', s=300, marker='s', label='仓库', zorder=3, edgecolors='darkblue', linewidth=2)
    
    # 添加节点标签
    for i, (x, y) in enumerate(coordinates):
        if demands[i] > 0:
            ax1.annotate(f'{i}', (x, y), xytext=(6, 6), textcoords='offset points', 
                        fontsize=9, fontweight='bold', color='darkred')
        else:
            ax1.annotate('仓库', (x, y), xytext=(6, 6), textcoords='offset points', 
                        fontsize=10, fontweight='bold', color='darkblue')
    
    ax1.set_xlabel('X 坐标')
    ax1.set_ylabel('Y 坐标')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # === 子图2: 解决方案路径 ===
    ax2.set_title('解决方案路径', fontsize=12, fontweight='bold')
    
    # 重新绘制节点
    ax2.scatter(coordinates[customer_indices, 0], coordinates[customer_indices, 1], 
               c='red', s=120, alpha=0.6, label='客户')
    ax2.scatter(coordinates[depot_idx, 0], coordinates[depot_idx, 1], 
               c='blue', s=300, marker='s', label='仓库', edgecolors='darkblue')
    
    if solution and 'truck_route' in solution:
        truck_route = solution['truck_route']
        
        # 绘制路径
        truck_coords = coordinates[truck_route]
        ax2.plot(truck_coords[:, 0], truck_coords[:, 1], 
                'g-', linewidth=3, label='卡车路径', alpha=0.9, zorder=1)
        
        # 添加方向箭头
        for i in range(len(truck_route) - 1):
            start = coordinates[truck_route[i]]
            end = coordinates[truck_route[i + 1]]
            dx, dy = end[0] - start[0], end[1] - start[1]
            
            # 计算箭头位置（在线段中点）
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            ax2.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.8))
        
        # 高亮路径节点
        route_coords = coordinates[truck_route]
        ax2.scatter(route_coords[:, 0], route_coords[:, 1], 
                   c='lime', s=100, alpha=0.8, label='路径节点', zorder=4, edgecolors='darkgreen')
    else:
        ax2.text(0.5, 0.5, '暂无解决方案', transform=ax2.transAxes, ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    
    ax2.set_xlabel('X 坐标')
    ax2.set_ylabel('Y 坐标')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # === 子图3: 问题统计 ===
    ax3.set_title('问题统计信息', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 计算统计信息
    customer_coords = coordinates[customer_indices]
    depot_coord = coordinates[depot_idx]
    
    # 计算到仓库的距离
    distances_to_depot = [np.linalg.norm(coord - depot_coord) for coord in customer_coords]
    
    stats_text = f"""问题详细信息:
    
节点总数: {n_nodes}
客户数量: {len(customer_indices)}
仓库位置: ({depot_coord[0]:.1f}, {depot_coord[1]:.1f})

需求分布:
• 总需求: {np.sum(demands):.1f}
• 平均需求: {np.mean(demands[demands > 0]):.2f}
• 最大需求: {np.max(demands):.1f}

几何分布:
• 最近客户距离: {min(distances_to_depot):.1f}
• 最远客户距离: {max(distances_to_depot):.1f}
• 平均距离: {np.mean(distances_to_depot):.1f}
• 距离标准差: {np.std(distances_to_depot):.1f}

卡车参数:
• 速度: 1.0 单位/时间
• 无人机速度: 2.0 单位/时间
• 电池续航: 150 时间单位
    """
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
    
    # === 子图4: 解决方案结果 ===
    ax4.set_title('解决方案结果', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    if solution:
        total_time = solution.get('total_time', 0)
        truck_visits = len(set(solution.get('truck_route', [])))
        reward = solution.get('reward', 0)
        
        # 计算路径统计
        route = solution.get('truck_route', [])
        if len(route) > 1:
            path_distances = []
            for i in range(len(route) - 1):
                dist = np.linalg.norm(coordinates[route[i]] - coordinates[route[i+1]])
                path_distances.append(dist)
            
            avg_step_distance = np.mean(path_distances)
            max_step_distance = max(path_distances)
            min_step_distance = min(path_distances)
        else:
            avg_step_distance = max_step_distance = min_step_distance = 0
        
        solution_text = f"""解决方案性能:
        
总时间成本: {total_time:.2f}
总奖励值: {reward:.2f}
卡车访问节点: {truck_visits} / {n_nodes}
路径效率: {truck_visits/n_nodes*100:.1f}%

路径分析:
• 平均步长: {avg_step_distance:.1f}
• 最大步长: {max_step_distance:.1f}
• 最小步长: {min_step_distance:.1f}
• 路径段数: {len(route)-1}

性能评级:
• 访问密度: {'优秀' if truck_visits/n_nodes > 0.8 else '良好' if truck_visits/n_nodes > 0.6 else '一般'}
• 路径长度: {'优秀' if avg_step_distance < 20 else '良好' if avg_step_distance < 35 else '一般'}
        """
    else:
        solution_text = """解决方案结果:
        
尚未求解
使用预训练模型
或贪心算法
获取最优路径

提示:
• 检查预训练模型
• 确保数据格式正确
• 调整算法参数
        """
    
    ax4.text(0.05, 0.95, solution_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图形已保存到: {save_path}")
    
    return fig

def run_quick_demo():
    """运行快速演示"""
    print("=" * 60)
    print("🚀 TSP Drone 快速演示 - 图形化可视化")
    print("=" * 60)
    
    # 1. 创建示例数据
    print("\n📊 1. 生成示例数据...")
    n_nodes = 11
    data = create_sample_data(n_nodes)
    print(f"   ✓ 创建了 {n_nodes} 节点的问题实例")
    print(f"   ✓ 包含 {n_nodes-1} 个客户和 1 个仓库")
    
    # 2. 尝试加载预训练模型
    print("\n🤖 2. 加载预训练模型...")
    actor, critic, config = load_pretrained_model(n_nodes)
    
    # 3. 求解问题
    print("\n🔍 3. 求解问题...")
    solution = None
    
    if actor is not None:
        print("   尝试使用预训练模型求解...")
        solution = solve_problem_with_model(data, actor, critic, config)
    
    if solution is None:
        print("   使用贪心算法求解...")
        solution = create_greedy_solution(data)
        print(f"   ✓ 贪心算法完成: 总时间 = {solution['total_time']:.2f}")
    
    # 4. 创建可视化
    print("\n🎨 4. 生成图形化可视化...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"quick_demo_{timestamp}.png"
    
    fig = visualize_tsp_drone_comprehensive(data, solution, save_path)
    
    print(f"\n✨ 演示完成!")
    print(f"📁 结果保存在: {save_path}")
    print("\n🔍 可视化内容包括:")
    print("   • 问题设置 - 显示客户和仓库位置")
    print("   • 解决方案路径 - 显示最优路径")
    print("   • 问题统计 - 详细的数据分析")
    print("   • 解决方案结果 - 性能评估")
    
    # 5. 显示图形
    print("\n🖼️  显示图形界面...")
    plt.show()
    
    return fig, data, solution

def main():
    """主函数"""
    import sys
    
    try:
        # 检查命令行参数
        if len(sys.argv) > 1:
            n_nodes = int(sys.argv[1])
        else:
            n_nodes = 11
        
        print(f"🎯 使用 {n_nodes} 节点进行演示")
        
        # 运行演示
        fig, data, solution = run_quick_demo()
        
        print("\n" + "=" * 60)
        print("💡 提示:")
        print("   • 可以通过修改 n_nodes 参数测试不同规模")
        print("   • 预训练模型位于 trained_models/ 目录")
        print("   • 使用 interactive_visualizer.py 获得交互式体验")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()