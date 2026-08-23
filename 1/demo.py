#!/usr/bin/env python3
"""
TSPDrone-RL 项目功能演示脚本

这个脚本演示了基于深度强化学习的无人机卡车协同路径规划系统的完整功能。
包括：
1. 多规模问题的求解和可视化
2. 解决方案质量分析
3. 算法性能对比
4. 完整的演示流程

使用方法:
    python demo.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import time
import argparse

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.env_no_comb import Env, DataGenerator
from utils.options import ParseParams
from model.nnets import Actor, Critic
from utils.agent import A2CAgent
import visualize_solution
import copy
import torch

class TSPDroneDemo:
    """TSPDrone-RL 项目演示类"""
    
    def __init__(self):
        self.results = []
        self.config = self.setup_config()
        
    def setup_config(self):
        """设置演示配置"""
        config = {
            'hidden_dim': 256,
            'decode_len': 20,
            'batch_size': 1,
            'random_seed': 42,
            'v_t': 1.0,  # 卡车速度
            'v_d': 2.0,  # 无人机速度
            'R': 150,    # 电池范围
            'n_nodes': 11,
            'test_size': 5,
            'data_dir': 'data',
            'save_path': 'trained_models'
        }
        return config
    
    def print_banner(self):
        """打印项目横幅"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         🚁 TSPDrone-RL 项目演示 🚁                           ║
║              基于深度强化学习的无人机卡车协同路径规划系统                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        
    def print_section(self, title):
        """打印章节标题"""
        print(f"\n{'='*80}")
        print(f" {title}")
        print(f"{'='*80}")
        
    def check_models(self):
        """检查预训练模型是否存在"""
        self.print_section("检查预训练模型")
        
        model_found = True
        for n_nodes in [11, 20]:
            actor_path = f"trained_models/n{n_nodes}/best_model_actor_truck_params.pkl"
            critic_path = f"trained_models/n{n_nodes}/best_model_critic_params.pkl"
            
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                print(f"✅ 找到 {n_nodes} 节点的预训练模型")
            else:
                print(f"❌ 未找到 {n_nodes} 节点的预训练模型")
                model_found = False
        
        return model_found
    
    def load_models(self, n_nodes):
        """加载指定规模的模型"""
        actor = Actor(self.config['hidden_dim'])
        critic = Critic(self.config['hidden_dim'])
        
        actor_path = f"trained_models/n{n_nodes}/best_model_actor_truck_params.pkl"
        critic_path = f"trained_models/n{n_nodes}/best_model_critic_params.pkl"
        
        if os.path.exists(actor_path):
            actor.load_state_dict(torch.load(actor_path, map_location='cpu'))
            print(f"✅ 成功加载演员模型: {actor_path}")
        
        if os.path.exists(critic_path):
            critic.load_state_dict(torch.load(critic_path, map_location='cpu'))
            print(f"✅ 成功加载评论家模型: {critic_path}")
        
        return actor, critic
    
    def solve_problem(self, actor, critic, problem_idx, n_nodes, max_problems=3):
        """求解单个问题"""
        try:
            # 创建配置
            config = self.config.copy()
            config['n_nodes'] = n_nodes
            
            # 加载数据
            dataGen = DataGenerator(config)
            test_data = dataGen.get_test_all()
            
            if problem_idx >= len(test_data):
                print(f"⚠️  问题索引 {problem_idx} 超出范围，使用索引 0")
                problem_idx = 0
                
            problem_data = test_data[problem_idx:problem_idx+1]
            
            # 创建环境
            env = Env(config, problem_data)
            dynamic, avail_actions = env.reset()
            
            # 设备
            device = torch.device("cpu")
            actor_device = copy.deepcopy(actor).to(device)
            
            # 初始化时间向量 - 在 torch.no_grad() 外部初始化
            batch_size = 1
            time_vec_truck = np.zeros([batch_size, 2])
            time_vec_drone = np.zeros([batch_size, 2])
            
            # 求解过程
            with torch.no_grad():
                static_input = torch.from_numpy(env.input_data[:, :, :2]).float().to(device)
                static_hidden = actor_device.attention_encoder(static_input)
                static_hidden = static_hidden.permute(0, 2, 1)
                
                state = env.state
                terminated = env.terminated
                
                decoder_input = static_hidden.new_zeros(batch_size, self.config['hidden_dim'], 1)
                last_hh = None
                
                truck_route = []
                current_time = 0
                
                for step in range(config['decode_len']):
                    if terminated.all():
                        break
                        
                    if len(state.shape) == 3:
                        state_for_input = state[:, :, 0]
                    else:
                        state_for_input = state
                        
                    dynamic_truck = torch.from_numpy(state_for_input).float().to(device).unsqueeze(-1)
                    avail_actions_truck = avail_actions[:, :, 0:1].squeeze(-1)
                    
                    idx_truck, prob, logp, last_hh = actor_device.forward(
                        static_hidden, dynamic_truck, decoder_input, last_hh, 
                        terminated, avail_actions_truck)
                    
                    truck_route.append(idx_truck.item())
                    
                    idx_expanded = idx_truck.view(-1, 1, 1).expand(batch_size, self.config['hidden_dim'], 1)
                    decoder_input = torch.gather(static_hidden, 2, idx_expanded).detach()
                    
                    idx_drone = env.drone_loc  # 保持与 visualize_solution.py 一致
                    state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(
                        idx_truck.cpu().numpy(), idx_drone, time_vec_truck, time_vec_drone, terminated)
                    terminated = ter
                    
                total_time = env.current_time[0]
            
            return {
                'truck_route': truck_route,
                'total_time': total_time,
                'problem_data': problem_data[0],
                'n_nodes': n_nodes,
                'problem_idx': problem_idx
            }
            
        except Exception as e:
            print(f"❌ 求解问题时出错: {e}")
            return None
    
    def visualize_solution(self, solution, output_dir="demo_results"):
        """可视化解决方案"""
        if solution is None:
            return None
            
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建配置对象用于可视化
            class Args:
                def __init__(self, config):
                    self.output_dir = output_dir
                    self.n_nodes = config['n_nodes']
                    self.problem_idx = solution['problem_idx']
            
            args = Args(self.config)
            save_path = os.path.join(output_dir, f'tsp_solution_{solution["n_nodes"]}_nodes_idx_{solution["problem_idx"]}.png')
            
            # 可视化
            fig = visualize_solution.visualize_solution_comprehensive(
                solution['problem_data'], solution, save_path, self.config)
            
            return save_path
            
        except Exception as e:
            print(f"❌ 可视化时出错: {e}")
            return None
    
    def analyze_solution(self, solution):
        """分析解决方案质量"""
        if solution is None:
            return {}
            
        coordinates = solution['problem_data'][:, :2]
        demands = solution['problem_data'][:, 2]
        
        # 计算路径长度
        truck_route = solution['truck_route']
        total_distance = 0
        
        for i in range(len(truck_route) - 1):
            node1, node2 = truck_route[i], truck_route[i + 1]
            distance = np.linalg.norm(coordinates[node1] - coordinates[node2])
            total_distance += distance
        
        # 计算指标
        n_customers = len(np.where(demands > 0)[0])
        depot_idx = len(coordinates) - 1
        
        # 检查路径合理性
        starts_at_depot = truck_route[0] == depot_idx
        ends_at_depot = truck_route[-1] == depot_idx
        
        analysis = {
            'total_time': solution['total_time'],
            'total_distance': total_distance,
            'route_length': len(truck_route),
            'n_customers': n_customers,
            'starts_at_depot': starts_at_depot,
            'ends_at_depot': ends_at_depot,
            'efficiency': total_distance / solution['total_time'] if solution['total_time'] > 0 else 0
        }
        
        return analysis
    
    def demo_single_scale(self, n_nodes, n_problems=3):
        """演示单个规模的求解"""
        self.print_section(f"演示 {n_nodes} 节点规模的 TSP 问题")
        
        if not self.check_models():
            print(f"⚠️  缺少 {n_nodes} 节点的预训练模型，跳过此规模演示")
            return []
        
        # 加载模型
        actor, critic = self.load_models(n_nodes)
        
        # 更新配置
        config = self.config.copy()
        config['n_nodes'] = n_nodes
        
        results = []
        
        for i in range(n_problems):
            print(f"\n🔍 求解问题 {i+1}/{n_problems} (规模: {n_nodes} 节点)...")
            
            start_time = time.time()
            solution = self.solve_problem(actor, critic, i, n_nodes)
            solve_time = time.time() - start_time
            
            if solution:
                # 分析解决方案
                analysis = self.analyze_solution(solution)
                analysis['solve_time'] = solve_time
                analysis['n_nodes'] = n_nodes
                analysis['problem_idx'] = i
                
                print(f"✅ 求解完成:")
                print(f"   🛣️  路径长度: {len(solution['truck_route'])} 步")
                print(f"   ⏱️  总时间: {solution['total_time']:.2f}")
                print(f"   📏 总距离: {analysis['total_distance']:.2f}")
                print(f"   🚛 从仓库出发: {'是' if analysis['starts_at_depot'] else '否'}")
                print(f"   🏠 回到仓库: {'是' if analysis['ends_at_depot'] else '否'}")
                print(f"   ⏳ 求解时间: {solve_time:.3f}s")
                
                # 可视化
                print(f"   🎨 生成可视化...")
                viz_path = self.visualize_solution(solution, f"demo_results/n{n_nodes}")
                
                results.append({
                    'solution': solution,
                    'analysis': analysis,
                    'visualization_path': viz_path
                })
            else:
                print(f"❌ 求解失败")
        
        return results
    
    def compare_scales(self, results_list):
        """对比不同规模的求解结果"""
        self.print_section("不同规模问题性能对比")
        
        print(f"{'规模':<8} {'问题数':<8} {'平均时间':<12} {'平均距离':<12} {'效率':<10}")
        print("-" * 60)
        
        for results in results_list:
            if not results:
                continue
                
            n_nodes = results[0]['analysis']['n_nodes']
            n_problems = len(results)
            
            avg_time = np.mean([r['analysis']['total_time'] for r in results])
            avg_distance = np.mean([r['analysis']['total_distance'] for r in results])
            avg_efficiency = np.mean([r['analysis']['efficiency'] for r in results])
            
            print(f"{n_nodes:<8} {n_problems:<8} {avg_time:<12.2f} {avg_distance:<12.2f} {avg_efficiency:<10.3f}")
    
    def demo_features(self):
        """演示项目功能特性"""
        self.print_section("项目功能特性介绍")
        
        features = [
            "🎯 深度强化学习: 使用 A2C 算法训练智能体",
            "🧠 注意力机制: Graph Attention Encoder 处理图结构",
            "🚁 协同规划: 卡车和无人机协同完成配送任务",
            "📊 可视化分析: 完整的解决方案可视化和分析",
            "⚡ 高效推理: 支持实时路径规划和决策",
            "📈 性能优化: 针对不同规模问题的优化",
            "🔧 模块化设计: 易于扩展和定制",
            "💾 预训练模型: 多个规模的预训练模型"
        ]
        
        for feature in features:
            print(f"  {feature}")
    
    def run_complete_demo(self):
        """运行完整演示"""
        self.print_banner()
        
        # 项目特性介绍
        self.demo_features()
        
        # 检查模型
        if not self.check_models():
            print("\n⚠️  未找到预训练模型。请确保模型文件存在于 trained_models/ 目录中。")
            return
        
        # 演示不同规模的问题
        scales = [11, 20]
        all_results = []
        
        for scale in scales:
            results = self.demo_single_scale(scale, n_problems=2)
            all_results.append(results)
        
        # 性能对比
        self.compare_scales(all_results)
        
        # 总结
        self.print_section("演示总结")
        
        total_solutions = sum(len(results) for results in all_results)
        print(f"✅ 成功演示了 {total_solutions} 个解决方案")
        print(f"📁 可视化结果保存在: demo_results/ 目录")
        print(f"🎯 涵盖了 {len(scales)} 种不同规模的问题")
        
        print(f"\n🚀 TSPDrone-RL 项目演示完成!")
        print(f"   您可以查看 demo_results/ 目录中的可视化结果")
        print(f"   或运行 python visualize_solution.py 进行单实例可视化")
    
    def interactive_demo(self):
        """交互式演示"""
        self.print_banner()
        
        if not self.check_models():
            print("❌ 未找到预训练模型，无法进行交互式演示")
            return
        
        print("\n🎮 交互式演示模式")
        print("输入 'quit' 退出程序")
        
        while True:
            try:
                print("\n" + "-" * 50)
                n_nodes = input("请输入问题规模 (11/20) [默认: 11]: ").strip()
                if n_nodes.lower() in ['quit', 'exit', 'q']:
                    break
                if not n_nodes:
                    n_nodes = 11
                else:
                    n_nodes = int(n_nodes)
                
                problem_idx = input("请输入问题索引 [默认: 0]: ").strip()
                if not problem_idx:
                    problem_idx = 0
                else:
                    problem_idx = int(problem_idx)
                
                if n_nodes not in [11, 20]:
                    print("❌ 只支持 11 和 20 节点规模")
                    continue
                
                # 求解
                actor, critic = self.load_models(n_nodes)
                solution = self.solve_problem(actor, critic, problem_idx, n_nodes)
                
                if solution:
                    analysis = self.analyze_solution(solution)
                    print(f"\n✅ 求解结果:")
                    print(f"   路径: {solution['truck_route']}")
                    print(f"   总时间: {analysis['total_time']:.2f}")
                    print(f"   总距离: {analysis['total_distance']:.2f}")
                    
                    # 询问是否可视化
                    viz = input("是否生成可视化? (y/n) [默认: y]: ").strip().lower()
                    if viz in ['', 'y', 'yes']:
                        viz_path = self.visualize_solution(solution, "interactive_results")
                        if viz_path:
                            print(f"   可视化已保存: {viz_path}")
                else:
                    print("❌ 求解失败")
                    
            except KeyboardInterrupt:
                print("\n\n👋 交互式演示结束")
                break
            except Exception as e:
                print(f"❌ 出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TSPDrone-RL 项目演示")
    parser.add_argument("--mode", choices=["auto", "interactive"], default="auto",
                       help="演示模式: auto=自动演示, interactive=交互式演示")
    
    args = parser.parse_args()
    
    demo = TSPDroneDemo()
    
    if args.mode == "interactive":
        demo.interactive_demo()
    else:
        demo.run_complete_demo()

if __name__ == "__main__":
    main()