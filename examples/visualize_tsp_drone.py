import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.utils.env_no_comb import Env, DataGenerator
from src.utils.options import ParseParams

def parse_solution_log(log_file):
    """解析解决方案日志，提取路径信息"""
    routes = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()
            # 这里需要根据实际日志格式来解析
            # 暂时返回一个示例
            pass
    return routes

def load_test_data(args, problem_idx=0):
    """加载测试数据"""
    dataGen = DataGenerator(args)
    test_data = dataGen.get_test_all()
    return test_data[problem_idx]

def visualize_single_problem(data, solution=None, save_path=None, title="TSP Drone Problem"):
    """可视化单个TSP Drone问题"""
    
    # 提取坐标和需求
    coordinates = data[:, :2]  # (n_nodes, 2)
    demands = data[:, 2]      # (n_nodes,)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：问题可视化
    ax1.set_title("TSP Drone Problem Setup", fontsize=14, fontweight='bold')
    
    # 绘制节点
    depot_idx = len(demands) - 1  # 最后一个节点是仓库
    customer_indices = np.where(demands > 0)[0]
    
    # 绘制客户节点
    ax1.scatter(coordinates[customer_indices, 0], coordinates[customer_indices, 1], 
               c='red', s=100, alpha=0.7, label='Customers')
    
    # 绘制仓库
    ax1.scatter(coordinates[depot_idx, 0], coordinates[depot_idx, 1], 
               c='blue', s=200, marker='s', label='Depot')
    
    # 添加节点标签
    for i, (x, y) in enumerate(coordinates):
        demand_text = f"({demands[i]:.1f})" if demands[i] > 0 else "(Depot)"
        ax1.annotate(f'{i}: {demand_text}', (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：解决方案可视化（如果有的话）
    ax2.set_title("Solution Visualization", fontsize=14, fontweight='bold')
    
    # 重新绘制节点
    ax2.scatter(coordinates[customer_indices, 0], coordinates[customer_indices, 1], 
               c='red', s=100, alpha=0.7, label='Customers')
    ax2.scatter(coordinates[depot_idx, 0], coordinates[depot_idx, 1], 
               c='blue', s=200, marker='s', label='Depot')
    
    if solution is not None:
        # 如果有解决方案，绘制路径
        truck_route = solution.get('truck_route', [])
        drone_routes = solution.get('drone_routes', [])
        
        # 绘制卡车路径
        if truck_route and len(truck_route) > 1:
            truck_coords = coordinates[truck_route]
            ax2.plot(truck_coords[:, 0], truck_coords[:, 1], 
                    'b-', linewidth=2, label='Truck Route', alpha=0.8)
            
            # 添加方向箭头
            for i in range(len(truck_route) - 1):
                start = coordinates[truck_route[i]]
                end = coordinates[truck_route[i + 1]]
                ax2.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
        
        # 绘制无人机路径
        colors = ['green', 'orange', 'purple', 'brown', 'pink']
        for i, drone_route in enumerate(drone_routes):
            if drone_route and len(drone_route) > 1:
                drone_coords = coordinates[drone_route]
                color = colors[i % len(colors)]
                ax2.plot(drone_coords[:, 0], drone_coords[:, 1], 
                        color=color, linewidth=2, label=f'Drone Route {i+1}', 
                        linestyle='--', alpha=0.8)
    else:
        ax2.text(0.5, 0.5, 'No solution available\nRun the TSP Drone solver first', 
                transform=ax2.transAxes, ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    return fig

def visualize_training_progress(log_file, save_path=None):
    """可视化训练进度"""
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found")
        return None
    
    # 解析训练日志
    epochs = []
    rewards = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'reward' in line.lower():
                # 简单的日志解析示例
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'Epoch' in part:
                        epoch = int(part.split(':')[1])
                        epochs.append(epoch)
                    elif 'reward' in part.lower():
                        reward = float(parts[i+1])
                        rewards.append(reward)
                        break
    
    if not epochs:
        print("No training data found in log file")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 奖励曲线
    ax1.plot(epochs, rewards, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress - Reward')
    ax1.grid(True, alpha=0.3)
    
    # 奖励分布
    ax2.hist(rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training visualization saved to: {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize TSP Drone Problems and Solutions')
    parser.add_argument('--problem_idx', type=int, default=0, help='Problem index to visualize')
    parser.add_argument('--n_nodes', type=int, default=11, help='Number of nodes')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory')
    parser.add_argument('--show_training', action='store_true', help='Show training progress')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    config = {
        'n_nodes': args.n_nodes,
        'batch_size': 1,
        'test_size': 1,
        'random_seed': 42,
        'data_dir': 'data'
    }
    
    try:
        # 可视化问题实例
        print(f"Visualizing problem {args.problem_idx} with {args.n_nodes} nodes...")
        data = load_test_data(config, args.problem_idx)
        
        # 创建可视化
        save_path = os.path.join(args.output_dir, f'tsp_problem_{args.n_nodes}_nodes_idx_{args.problem_idx}.png')
        fig = visualize_single_problem(data, save_path=save_path)
        
        # 可视化训练进度（如果需要）
        if args.show_training:
            log_path = 'logs/results.txt'
            training_save_path = os.path.join(args.output_dir, 'training_progress.png')
            visualize_training_progress(log_path, training_save_path)
        
        print(f"Visualization completed!")
        
        # 显示图像
        plt.show()
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()