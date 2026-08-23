import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import torch
import argparse
import copy

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.utils.env_no_comb import Env, DataGenerator
from src.utils.options import ParseParams
from src.model.nnets import Actor, Critic
from src.utils.agent import A2CAgent

def load_trained_model(args, n_nodes):
    """加载训练好的模型"""
    actor = Actor(args['hidden_dim'])
    critic = Critic(args['hidden_dim'])
    
    save_path = args['save_path']
    actor_path = os.path.join(save_path, f'n{n_nodes}', 'best_model_actor_truck_params.pkl')
    critic_path = os.path.join(save_path, f'n{n_nodes}', 'best_model_critic_params.pkl')
    
    if os.path.exists(actor_path):
        actor.load_state_dict(torch.load(actor_path, map_location='cpu'))
        print(f"Successfully loaded actor model from {actor_path}")
    
    if os.path.exists(critic_path):
        critic.load_state_dict(torch.load(critic_path, map_location='cpu'))
        print(f"Successfully loaded critic model from {critic_path}")
    
    return actor, critic

def solve_single_problem(actor, critic, args, problem_idx=0):
    """为单个问题求解最优解"""
    # 加载数据
    dataGen = DataGenerator(args)
    test_data = dataGen.get_test_all()
    problem_data = test_data[problem_idx:problem_idx+1]  # 取单个问题
    
    # 创建环境
    env = Env(args, problem_data)
    dynamic, avail_actions = env.reset()
    
    # 设备
    device = torch.device("cpu")  # 使用CPU避免CUDA问题
    
    # 复制模型到设备
    actor_device = copy.deepcopy(actor).to(device)
    critic_device = copy.deepcopy(critic).to(device)
    
    # 创建智能体
    agent = A2CAgent(actor_device, critic_device, args, env, dataGen)
    
    # 求解（使用贪婪策略）
    with torch.no_grad():
        # 获取初始状态
        # 使用正确的输入格式 [batch_size, n_nodes, 2]
        static_input = torch.from_numpy(env.input_data[:, :, :2]).float().to(device)
        # 通过Actor的attention encoder获取高维特征
        static_hidden = actor_device.attention_encoder(static_input)
        # 转换为卷积层期望的格式 [batch_size, hidden_dim, n_nodes]
        static_hidden = static_hidden.permute(0, 2, 1)
        state = env.state
        terminated = env.terminated
        
        # 初始化解码器输入
        batch_size = 1
        # 修改维度为三维 [batch_size, hidden_dim, 1]
        decoder_input = static_hidden.new_zeros(batch_size, args['hidden_dim'], 1)
        last_hh = None
        
        # 记录路径
        truck_route = []
        drone_routes = []
        current_drone_route = []
        time_vec_truck = np.zeros([batch_size, 2])
        time_vec_drone = np.zeros([batch_size, 2])
        
        # 决策过程
        for step in range(args['decode_len']):
            if terminated.all():
                break
                
            # 卡车决策
            # 使用正确的维度访问动态状态
            # 确保state是正确的格式 [batch_size, n_nodes]
            if len(state.shape) == 3:  # 如果state是[batch_size, n_nodes, something]，只取前两维
                state_for_input = state[:, :, 0]  # 取第一个特征作为动态输入
            else:
                state_for_input = state
            dynamic_truck = torch.from_numpy(state_for_input).float().to(device).unsqueeze(-1)  # [batch_size, n_nodes, 1]
            # 只使用卡车相关的可用操作 [batch_size, n_nodes, 1]
            avail_actions_truck = avail_actions[:, :, 0:1].squeeze(-1)  # [batch_size, n_nodes]
            idx_truck, prob, logp, last_hh = actor_device.forward(
                static_hidden, dynamic_truck, decoder_input, last_hh, 
                terminated, avail_actions_truck)
            
            truck_route.append(idx_truck.item())
            
            # 更新状态
            # 修改为保持维度一致
            idx_expanded = idx_truck.view(-1, 1, 1).expand(batch_size, args['hidden_dim'], 1)
            decoder_input = torch.gather(static_hidden, 2, idx_expanded).detach()
            
            # 环境更新
            # 在这个简单的可视化中，我们只关心卡车路径，所以无人机决策使用当前位置
            idx_drone = env.drone_loc  # 无人机保持在当前位置
            state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(
                idx_truck.cpu().numpy(), idx_drone, time_vec_truck, time_vec_drone, terminated)
            terminated = ter
            
        total_time = env.current_time[0]
        
    return {
        'truck_route': truck_route,
        'total_time': total_time,
        'problem_data': problem_data[0]
    }

def visualize_solution_comprehensive(data, solution, save_path=None, config=None):
    """综合可视化TSP Drone解决方案"""
    
    coordinates = data[:, :2]  # (n_nodes, 2)
    demands = data[:, 2]      # (n_nodes,)
    
    # 如果没有提供配置，使用默认配置
    if config is None:
        config = {
            'v_t': 1.0,  # 默认卡车速度
            'v_d': 2.0,  # 默认无人机速度
            'R': 150     # 默认电池范围
        }
    
    n_nodes = len(coordinates)
    depot_idx = n_nodes - 1  # 最后一个节点是仓库
    
    # 创建大图
    fig = plt.figure(figsize=(20, 12))
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1])
    
    # 主可视化区域
    ax_main = fig.add_subplot(gs[:2, :2])
    
    # 问题信息区域
    ax_info = fig.add_subplot(gs[0, 2])
    
    # 统计信息区域
    ax_stats = fig.add_subplot(gs[1, 2])
    
    # 时间线区域
    ax_timeline = fig.add_subplot(gs[2, :])
    
    # ===== 主可视化 =====
    ax_main.set_title("TSP Drone Solution Visualization", fontsize=16, fontweight='bold', pad=20)
    
    # 绘制节点
    customer_indices = np.where(demands > 0)[0]
    
    # 绘制所有节点
    ax_main.scatter(coordinates[customer_indices, 0], coordinates[customer_indices, 1], 
                   c='red', s=150, alpha=0.8, label='Customers', zorder=3, edgecolors='darkred')
    
    ax_main.scatter(coordinates[depot_idx, 0], coordinates[depot_idx, 1], 
                   c='blue', s=300, marker='s', label='Depot', zorder=3, edgecolors='darkblue')
    
    # 添加节点标签
    for i, (x, y) in enumerate(coordinates):
        if demands[i] > 0:
            ax_main.annotate(f'{i}\n(d={demands[i]:.1f})', (x, y), 
                           xytext=(8, 8), textcoords='offset points', 
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax_main.annotate(f'{i}\n(Depot)', (x, y), 
                           xytext=(8, 8), textcoords='offset points', 
                           fontsize=10, fontweight='bold', color='blue',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # 绘制卡车路径
    truck_route = solution.get('truck_route', [])
    if truck_route and len(truck_route) > 1:
        truck_coords = coordinates[truck_route]
        ax_main.plot(truck_coords[:, 0], truck_coords[:, 1], 
                    'b-', linewidth=4, label='Truck Route', alpha=0.9, zorder=1)
        
        # 添加方向箭头
        for i in range(len(truck_route) - 1):
            start = coordinates[truck_route[i]]
            end = coordinates[truck_route[i + 1]]
            dx, dy = end[0] - start[0], end[1] - start[1]
            ax_main.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.8))
    
    # 设置坐标轴
    ax_main.set_xlabel('X Coordinate', fontsize=12)
    ax_main.set_ylabel('Y Coordinate', fontsize=12)
    ax_main.legend(loc='upper right', fontsize=11)
    ax_main.grid(True, alpha=0.4)
    
    # ===== 问题信息 =====
    ax_info.set_title("Problem Information", fontsize=12, fontweight='bold')
    ax_info.axis('off')
    
    info_text = f"""Problem Details:
    
Nodes: {n_nodes}
Customers: {len(customer_indices)}
Depot: Node {depot_idx}

Total Demand: {np.sum(demands):.1f}
Avg Demand: {np.mean(demands[demands > 0]):.2f}
Max Demand: {np.max(demands):.1f}

Truck Speed: {config.get('v_t', 1)} units/time
Drone Speed: {config.get('v_d', 2)} units/time
Battery Range: {config.get('R', 150)} time units
    """
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # ===== 统计信息 =====
    ax_stats.set_title("Solution Statistics", fontsize=12, fontweight='bold')
    ax_stats.axis('off')
    
    total_time = solution.get('total_time', 0)
    truck_visits = len(set(truck_route))
    
    stats_text = f"""Solution Results:
    
Total Time: {total_time:.2f}
Truck Visits: {truck_visits}
Route Length: {len(truck_route)}

Nodes per Visit: {truck_visits/n_nodes*100:.1f}%
Efficiency: {n_nodes/truck_visits:.2f}
    """
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    # ===== 时间线可视化 =====
    ax_timeline.set_title("Execution Timeline", fontsize=12, fontweight='bold')
    ax_timeline.set_xlabel('Time')
    ax_timeline.set_ylabel('Vehicle')
    
    # 简化的活动时间线
    timeline_data = []
    current_time = 0
    
    # 基于路径估算时间（简化）
    for i, node in enumerate(truck_route[:-1]):  # 排除最后的仓库
        if node != depot_idx:  # 不是仓库
            # 模拟访问时间
            visit_duration = np.random.uniform(2, 5)  # 假设每个客户访问时间2-5单位
            timeline_data.append({
                'start': current_time,
                'end': current_time + visit_duration,
                'node': node,
                'type': 'Customer Visit'
            })
            current_time += visit_duration
        
        # 移动到下一个节点
        if i < len(truck_route) - 2:
            next_node = truck_route[i + 1]
            distance = np.linalg.norm(coordinates[node] - coordinates[next_node])
            travel_time = distance / config.get('v_t', 1)
            timeline_data.append({
                'start': current_time,
                'end': current_time + travel_time,
                'node': f"{node}→{next_node}",
                'type': 'Truck Travel'
            })
            current_time += travel_time
    
    # 绘制时间线
    colors = {'Customer Visit': 'red', 'Truck Travel': 'blue'}
    y_pos = 0
    
    for i, activity in enumerate(timeline_data[:10]):  # 只显示前10个活动
        color = colors.get(activity['type'], 'gray')
        ax_timeline.barh(y_pos, activity['end'] - activity['start'], 
                        left=activity['start'], height=0.6,
                        color=color, alpha=0.7, 
                        label=activity['type'] if i == 0 else "")
        
        # 添加标签
        mid_time = (activity['start'] + activity['end']) / 2
        ax_timeline.text(mid_time, y_pos, f"{activity['node']}", 
                        ha='center', va='center', fontsize=8)
        
        y_pos += 1
    
    ax_timeline.set_yticks(range(min(len(timeline_data), 10)))
    ax_timeline.set_yticklabels([f"Step {i+1}" for i in range(min(len(timeline_data), 10))])
    ax_timeline.legend()
    ax_timeline.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive visualization saved to: {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize TSP Drone Solutions')
    parser.add_argument('--problem_idx', type=int, default=0, help='Problem index to solve and visualize')
    parser.add_argument('--n_nodes', type=int, default=11, help='Number of nodes')
    parser.add_argument('--output_dir', type=str, default='solution_visualizations', help='Output directory')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    config = {
        'n_nodes': args.n_nodes,
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
        'stdout_print': False
    }
    
    try:
        print(f"Solving and visualizing problem {args.problem_idx} with {args.n_nodes} nodes...")
        
        # 加载训练好的模型
        actor, critic = load_trained_model(config, args.n_nodes)
        
        # 求解问题
        solution = solve_single_problem(actor, critic, config, args.problem_idx)
        
        # 可视化解决方案
        save_path = os.path.join(args.output_dir, f'tsp_solution_{args.n_nodes}_nodes_idx_{args.problem_idx}.png')
        fig = visualize_solution_comprehensive(solution['problem_data'], solution, save_path, config)
        
        print(f"Solution visualization completed!")
        print(f"Total time: {solution['total_time']:.2f}")
        print(f"Truck route: {solution['truck_route']}")
        
        # 显示图像
        plt.show()
        
    except Exception as e:
        print(f"Error during solution visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import torch
    main()