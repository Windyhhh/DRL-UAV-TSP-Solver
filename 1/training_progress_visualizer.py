#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSP Drone Training Progress Visualization Tool

This script is designed to visualize the training progress of deep reinforcement learning,
including loss functions, reward functions, success rates, and other key metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import argparse
from datetime import datetime
import seaborn as sns
from pathlib import Path

# Set English font to avoid Chinese font issues
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set chart style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingProgressVisualizer:
    """Training Progress Visualizer"""
    
    def __init__(self, data_dir=".", output_dir="training_visualizations"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Training data storage
        self.training_data = {
            'epochs': [],
            'rewards': [],
            'losses': [],
            'val_rewards': [],
            'learning_rates': []
        }
    
    def safe_get_array_data(self, key):
        """安全地获取数组数据，处理标量和其他异常情况"""
        data = self.training_data.get(key)
        if data is None:
            return None
        
        try:
            # 检查是否是数组类型
            if hasattr(data, '__len__') and not isinstance(data, str):
                if len(data) > 0:
                    return data
            # 如果是标量numpy数组
            elif hasattr(data, 'shape') and data.shape == ():
                return None
            else:
                return None
        except Exception:
            return None
        
    def load_existing_data(self):
        """Load existing training data"""
        print("Loading training data...")
        
        # 1. Load batch sampling reward data
        sampling_rewards_file = self.data_dir / "results" / "best_rewards_list_5_samples.txt"
        if sampling_rewards_file.exists():
            rewards_data = np.loadtxt(sampling_rewards_file)
            print(f"✓ Loaded batch sampling reward data: {rewards_data.shape}")
            self.training_data['sampling_rewards'] = rewards_data
        else:
            print("⚠ Batch sampling reward data not found")
            self.training_data['sampling_rewards'] = None
        
        # 2. Try to load test reward data
        test_rewards_file = self.data_dir / "trained_models" / "test_rewards.txt"
        if test_rewards_file.exists():
            test_rewards = np.loadtxt(test_rewards_file)
            print(f"✓ Loaded test reward data: {test_rewards.shape}")
            self.training_data['test_rewards'] = test_rewards
        else:
            print("⚠ Test reward data not found")
            self.training_data['test_rewards'] = None
        
        # 3. Load results log
        results_file = self.data_dir / "logs" / "results.txt"
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"✓ Loaded results log: {len(content)} characters")
                self.training_data['log_content'] = content
        else:
            print("⚠ Results log not found")
            self.training_data['log_content'] = ""
    
    def parse_training_logs(self):
        """Parse training logs to extract training metrics"""
        if not self.training_data.get('log_content'):
            print("No training log content available for parsing")
            return
        
        log_content = self.training_data['log_content']
        lines = log_content.split('\n')
        
        # Parse key information from logs
        epochs = []
        rewards = []
        losses = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to extract epoch information
            if 'epochs:' in line or 'Epoch' in line:
                try:
                    # Extract numbers
                    parts = line.split()
                    for part in parts:
                        if part.replace('.', '').isdigit():
                            epoch = float(part)
                            epochs.append(epoch)
                            break
                except:
                    continue
            
            # Try to extract reward information
            if 'reward' in line.lower() or 'time' in line.lower():
                try:
                    # Simple number extraction
                    import re
                    numbers = re.findall(r'[\d\.]+', line)
                    if numbers:
                        reward = float(numbers[0])
                        rewards.append(reward)
                except:
                    continue
        
        self.training_data['parsed_epochs'] = epochs
        self.training_data['parsed_rewards'] = rewards
        print(f"Parsed {len(epochs)} training epochs, {len(rewards)} reward values")
    
    def create_sample_training_data(self):
        """Create sample training data for demonstration (if no real data)"""
        print("Creating sample training data...")
        
        # Generate simulated training data
        n_epochs = 1000
        
        # Simulate reward curve (decreases then converges)
        initial_reward = 300
        final_reward = 180
        reward_noise = np.random.normal(0, 20, n_epochs)
        rewards = final_reward + (initial_reward - final_reward) * np.exp(-np.linspace(0, 3, n_epochs)) + reward_noise
        
        # Simulate loss curve
        initial_loss = 2.5
        final_loss = 0.8
        loss_noise = np.random.normal(0, 0.3, n_epochs)
        losses = final_loss + (initial_loss - final_loss) * np.exp(-np.linspace(0, 2.5, n_epochs)) + loss_noise
        
        # Simulate validation rewards
        val_rewards = rewards + np.random.normal(0, 10, n_epochs)
        
        # Simulate learning rate
        learning_rates = [1e-4 * (0.99 ** (i // 100)) for i in range(n_epochs)]
        
        self.training_data['sample_epochs'] = list(range(n_epochs))
        self.training_data['sample_rewards'] = rewards
        self.training_data['sample_losses'] = losses
        self.training_data['sample_val_rewards'] = val_rewards
        self.training_data['sample_learning_rates'] = learning_rates
    
    def plot_comprehensive_training_progress(self):
        """Plot comprehensive training progress charts"""
        print("Generating comprehensive training progress charts...")
        
        # Create large figure
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('TSP Drone Deep RL Training Progress Analysis', fontsize=20, fontweight='bold')
        
        # 创建子图网格
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. 奖励曲线
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_reward_curves(ax1)
        
        # 2. 奖励分布
        ax2 = fig.add_subplot(gs[0, 2])
        self.plot_reward_distribution(ax2)
        
        # 3. 损失曲线
        ax3 = fig.add_subplot(gs[1, :2])
        self.plot_loss_curves(ax3)
        
        # 4. 学习率变化
        ax4 = fig.add_subplot(gs[1, 2])
        self.plot_learning_rate(ax4)
        
        # 5. 性能对比
        ax5 = fig.add_subplot(gs[2, :])
        self.plot_performance_comparison(ax5)
        
        # 6. 训练统计摘要
        ax6 = fig.add_subplot(gs[3, :])
        self.plot_training_summary(ax6)
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = self.output_dir / f'comprehensive_training_progress_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ 综合训练进度图表已保存: {save_path}")
        
        return fig, save_path
    
    def plot_reward_curves(self, ax):
        """绘制奖励曲线"""
        # 尝试绘制真实数据
        has_real_data = False
        
        # Use safe data extraction
        test_rewards = self.safe_get_array_data('test_rewards')
        if test_rewards is not None:
            epochs = list(range(len(test_rewards)))
            ax.plot(epochs, test_rewards, 'b-', linewidth=2, label='Test Rewards', alpha=0.8)
            has_real_data = True
        
        sampling_rewards = self.safe_get_array_data('sampling_rewards')
        if sampling_rewards is not None:
            epochs = list(range(len(sampling_rewards)))
            ax.plot(epochs, sampling_rewards, 'g-', linewidth=2, label='Sampling Rewards', alpha=0.8)
            has_real_data = True
        
        # 如果没有真实数据，绘制示例数据
        if not has_real_data:
            if hasattr(self.training_data, 'sample_rewards'):
                epochs = self.training_data['sample_epochs']
                rewards = self.training_data['sample_rewards']
                ax.plot(epochs, rewards, 'r--', linewidth=2, label='Sample Training Rewards', alpha=0.7)
                
                # 添加置信区间
                if hasattr(self.training_data, 'sample_val_rewards'):
                    val_rewards = self.training_data['sample_val_rewards']
                    ax.fill_between(epochs, rewards, val_rewards, alpha=0.2, label='Confidence Range')
        
        ax.set_title('Reward Change Curves', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Epochs')
        ax.set_ylabel('Reward Value (Time Cost)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_reward_distribution(self, ax):
        """绘制奖励分布"""
        rewards_data = []
        
        # Collect all reward data
        test_rewards = self.safe_get_array_data('test_rewards')
        if test_rewards is not None:
            rewards_data.extend(test_rewards)
        
        sampling_rewards = self.safe_get_array_data('sampling_rewards')
        if sampling_rewards is not None:
            rewards_data.extend(sampling_rewards)
        
        if not rewards_data and hasattr(self.training_data, 'sample_rewards'):
            rewards_data = self.training_data['sample_rewards']
        
        if rewards_data:
            ax.hist(rewards_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title('Reward Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Reward Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistical information
            mean_reward = np.mean(rewards_data)
            std_reward = np.std(rewards_data)
            ax.axvline(mean_reward, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_reward:.1f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Reward Data', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_title('Reward Distribution', fontsize=14, fontweight='bold')
    
    def plot_loss_curves(self, ax):
        """绘制损失曲线"""
        if hasattr(self.training_data, 'sample_losses'):
            epochs = self.training_data['sample_epochs']
            losses = self.training_data['sample_losses']
            
            ax.plot(epochs, losses, 'orange', linewidth=2, label='Training Loss', alpha=0.8)
            ax.set_title('Loss Change Curves', fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Epochs')
            ax.set_ylabel('Loss Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add smoothed line
            window_size = 50
            if len(losses) > window_size:
                smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                smoothed_epochs = epochs[window_size-1:]
                ax.plot(smoothed_epochs, smoothed_losses, 'red', 
                       linewidth=3, label=f'Smoothed Loss (window={window_size})', alpha=0.9)
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Loss Data\n(Training logs required)', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_title('Loss Change Curves', fontsize=14, fontweight='bold')
    
    def plot_learning_rate(self, ax):
        """绘制学习率变化"""
        if hasattr(self.training_data, 'sample_learning_rates'):
            epochs = self.training_data['sample_epochs']
            learning_rates = self.training_data['sample_learning_rates']
            
            ax.plot(epochs, learning_rates, 'purple', linewidth=2, label='Learning Rate')
            ax.set_title('Learning Rate Changes', fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Epochs')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Learning Rate Data', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_title('Learning Rate Changes', fontsize=14, fontweight='bold')
    
    def plot_performance_comparison(self, ax):
        """绘制性能对比"""
        # Create comparison of different metrics
        metrics = ['Test Rewards', 'Sampling Rewards', 'Average Performance']
        values = []
        
        # Use safe data extraction for test rewards
        test_rewards = self.safe_get_array_data('test_rewards')
        if test_rewards is not None:
            values.append(np.mean(test_rewards))
        else:
            values.append(0)
        
        # Use safe data extraction for sampling rewards
        sampling_rewards = self.safe_get_array_data('sampling_rewards')
        if sampling_rewards is not None:
            values.append(np.mean(sampling_rewards))
        else:
            values.append(0)
        
        # Calculate comprehensive performance score
        if values[0] > 0 or values[1] > 0:
            values.append(np.mean([v for v in values if v > 0]))
        else:
            values.append(0)
        
        colors = ['skyblue', 'lightgreen', 'gold']
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Performance Comparison by Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Reward Value')
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_training_summary(self, ax):
        """绘制训练统计摘要"""
        ax.axis('off')
        
        # Calculate statistics
        summary_text = "Training Progress Statistical Summary:\n\n"
        
        # Reward statistics - use safe data extraction
        test_rewards = self.safe_get_array_data('test_rewards')
        if test_rewards is not None:
            summary_text += f"Test Reward Statistics:\n"
            summary_text += f"  • Mean: {np.mean(test_rewards):.2f}\n"
            summary_text += f"  • Std Dev: {np.std(test_rewards):.2f}\n"
            summary_text += f"  • Min: {np.min(test_rewards):.2f}\n"
            summary_text += f"  • Max: {np.max(test_rewards):.2f}\n"
            summary_text += f"  • Data Points: {len(test_rewards)}\n\n"
        
        sampling_rewards = self.safe_get_array_data('sampling_rewards')
        if sampling_rewards is not None:
            summary_text += f"Sampling Reward Statistics:\n"
            summary_text += f"  • Mean: {np.mean(sampling_rewards):.2f}\n"
            summary_text += f"  • Std Dev: {np.std(sampling_rewards):.2f}\n"
            summary_text += f"  • Data Points: {len(sampling_rewards)}\n\n"
        
        # Data availability
        summary_text += "Data Availability:\n"
        summary_text += f"  • Test Rewards: {'✓' if self.training_data.get('test_rewards') is not None else '✗'}\n"
        summary_text += f"  • Sampling Rewards: {'✓' if self.training_data.get('sampling_rewards') is not None else '✗'}\n"
        summary_text += f"  • Training Logs: {'✓' if self.training_data.get('log_content') else '✗'}\n"
        
        # Generation time
        summary_text += f"\nChart Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    def create_detailed_reports(self):
        """Create detailed analysis report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate text report
        report_path = self.output_dir / f'training_analysis_report_{timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("TSP Drone Training Progress Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Report Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Analyze reward data - use safe data extraction
            test_rewards = self.safe_get_array_data('test_rewards')
            if test_rewards is not None:
                f.write("Test Reward Analysis:\n")
                f.write(f"  Data Points: {len(test_rewards)}\n")
                f.write(f"  Mean Reward: {np.mean(test_rewards):.2f}\n")
                f.write(f"  Std Dev: {np.std(test_rewards):.2f}\n")
                f.write(f"  Min: {np.min(test_rewards):.2f}\n")
                f.write(f"  Max: {np.max(test_rewards):.2f}\n")
                f.write(f"  Median: {np.median(test_rewards):.2f}\n\n")
            
            sampling_rewards = self.safe_get_array_data('sampling_rewards')
            if sampling_rewards is not None:
                f.write("Sampling Reward Analysis:\n")
                f.write(f"  Data Points: {len(sampling_rewards)}\n")
                f.write(f"  Mean Reward: {np.mean(sampling_rewards):.2f}\n")
                f.write(f"  Std Dev: {np.std(sampling_rewards):.2f}\n")
                f.write(f"  Min: {np.min(sampling_rewards):.2f}\n")
                f.write(f"  Max: {np.max(sampling_rewards):.2f}\n\n")
            
            # Performance evaluation
            f.write("Performance Evaluation:\n")
            test_rewards = self.safe_get_array_data('test_rewards')
            sampling_rewards = self.safe_get_array_data('sampling_rewards')
            
            if test_rewards is not None and sampling_rewards is not None:
                test_mean = np.mean(test_rewards)
                sample_mean = np.mean(sampling_rewards)
                f.write(f"  Test vs Sampling Performance Difference: {abs(test_mean - sample_mean):.2f}\n")
                
                if test_mean < sample_mean:
                    f.write("  Conclusion: Test performance is better than sampling performance\n")
                else:
                    f.write("  Conclusion: Sampling performance is better than test performance\n")
            
            f.write("\nRecommendations:\n")
            f.write("1. If reward fluctuation is large, consider adjusting learning rate\n")
            f.write("2. If rewards continue to decrease, overfitting may be occurring\n")
            f.write("3. Recommend using early stopping to prevent overfitting\n")
            f.write("4. Can try different network architectures\n")
        
        print(f"✓ Detailed analysis report saved: {report_path}")
        return report_path

def main():
    parser = argparse.ArgumentParser(description='TSP Drone Training Progress Visualization Tool')
    parser.add_argument('--data_dir', type=str, default='.', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='training_visualizations', help='Output directory')
    parser.add_argument('--create_sample', action='store_true', help='Create sample data for demonstration')
    
    args = parser.parse_args()
    
    print("🚀 TSP Drone Training Progress Visualization Tool")
    print("=" * 60)
    
    # Create visualizer
    visualizer = TrainingProgressVisualizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # 加载数据
    visualizer.load_existing_data()
    
    # 解析训练日志
    visualizer.parse_training_logs()
    
    # 如果需要，创建示例数据
    if args.create_sample or (not visualizer.training_data.get('test_rewards') and 
                            not visualizer.training_data.get('sampling_rewards')):
        print("创建示例训练数据用于演示...")
        visualizer.create_sample_training_data()
    
    # 生成可视化
    print("\n📊 生成综合训练进度图表...")
    fig, save_path = visualizer.plot_comprehensive_training_progress()
    
    # 生成详细报告
    print("\n📋 生成详细分析报告...")
    report_path = visualizer.create_detailed_reports()
    
    print(f"\n✅ 训练进度可视化完成!")
    print(f"📁 图表保存位置: {save_path}")
    print(f"📋 报告保存位置: {report_path}")
    print(f"📂 所有文件保存在: {args.output_dir} 目录")
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    main()