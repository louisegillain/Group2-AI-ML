import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class EpisodeAnalyzer:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(csv_path)
        self._validate_data()
        
    def _validate_data(self):
        required = ['episode', 'steps', 'reward_sum']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def compute_rolling_metrics(self, window: int = 100) -> pd.DataFrame:
        df = self.df.copy()
        
        # Rolling statistics
        df['reward_rolling_mean'] = df['reward_sum'].rolling(window, min_periods=1).mean()
        df['reward_rolling_std'] = df['reward_sum'].rolling(window, min_periods=1).std()
        df['steps_rolling_mean'] = df['steps'].rolling(window, min_periods=1).mean()
        
        if 'accuracy_pct' in df.columns:
            df['accuracy_rolling_mean'] = df['accuracy_pct'].rolling(window, min_periods=1).mean()
        
        if 'success' in df.columns:
            df['success_rate'] = df['success'].rolling(window, min_periods=1).mean() * 100
        
        return df
    
    def get_performance_metrics(self) -> Dict[str, float]:
        metrics = {
            'total_episodes': len(self.df),
            'avg_reward': self.df['reward_sum'].mean(),
            'std_reward': self.df['reward_sum'].std(),
            'max_reward': self.df['reward_sum'].max(),
            'min_reward': self.df['reward_sum'].min(),
            'avg_steps': self.df['steps'].mean(),
            'std_steps': self.df['steps'].std()
        }
        
        # Optional metrics
        if 'accuracy_pct' in self.df.columns:
            metrics['avg_accuracy'] = self.df['accuracy_pct'].mean()
            metrics['std_accuracy'] = self.df['accuracy_pct'].std()
            metrics['final_accuracy'] = self.df['accuracy_pct'].tail(100).mean()
        
        if 'success' in self.df.columns:
            metrics['success_rate'] = self.df['success'].mean() * 100
            metrics['final_success_rate'] = self.df['success'].tail(100).mean() * 100
        
        if 'correct_placements' in self.df.columns:
            metrics['avg_correct_placements'] = self.df['correct_placements'].mean()
        
        if 'wrong_placements' in self.df.columns:
            metrics['avg_wrong_placements'] = self.df['wrong_placements'].mean()
        
        return metrics
    
    def find_convergence_point(self, 
                               target_metric: str = 'accuracy_pct',
                               threshold: float = 80.0,
                               window: int = 100) -> Optional[int]:
 
        if target_metric not in self.df.columns:
            return None
        
        rolling = self.df[target_metric].rolling(window, min_periods=window).mean()
        converged = rolling >= threshold
        
        if converged.any():
            return self.df.loc[converged.idxmax(), 'episode']
        return None
    
    def get_quartile_analysis(self) -> pd.DataFrame:
        n = len(self.df)
        self.df['quartile'] = pd.qcut(self.df['episode'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        quartile_stats = self.df.groupby('quartile').agg({
            'reward_sum': ['mean', 'std', 'min', 'max'],
            'steps': ['mean', 'std'],
            'accuracy_pct': 'mean' if 'accuracy_pct' in self.df.columns else lambda x: None
        }).round(2)
        
        return quartile_stats
    
    def plot_learning_curves(self, save_path: Optional[str] = None):
        df_rolling = self.compute_rolling_metrics(window=100)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Learning Curves - {self.csv_path.stem}', fontsize=16, fontweight='bold')
        
        # cumulative reward
        ax = axes[0, 0]
        ax.plot(df_rolling['episode'], df_rolling['reward_sum'], alpha=0.3, label='Raw', linewidth=0.5)
        ax.plot(df_rolling['episode'], df_rolling['reward_rolling_mean'], 
                label='Rolling Mean (100 ep)', linewidth=2, color='blue')
        ax.fill_between(df_rolling['episode'],
                        df_rolling['reward_rolling_mean'] - df_rolling['reward_rolling_std'],
                        df_rolling['reward_rolling_mean'] + df_rolling['reward_rolling_std'],
                        alpha=0.2, color='blue')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Reward Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # episode length
        ax = axes[0, 1]
        ax.plot(df_rolling['episode'], df_rolling['steps'], alpha=0.3, linewidth=0.5)
        ax.plot(df_rolling['episode'], df_rolling['steps_rolling_mean'], 
                linewidth=2, color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length Over Time')
        ax.grid(True, alpha=0.3)
        
        # accuracy 
        ax = axes[1, 0]
        if 'accuracy_pct' in df_rolling.columns:
            ax.plot(df_rolling['episode'], df_rolling['accuracy_pct'], alpha=0.3, linewidth=0.5)
            ax.plot(df_rolling['episode'], df_rolling['accuracy_rolling_mean'], 
                    linewidth=2, color='orange')
            ax.axhline(y=80, color='r', linestyle='--', label='80% Target')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Sorting Accuracy Over Time')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Accuracy data not available', 
                   ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)
        
        # success rate
        ax = axes[1, 1]
        if 'success_rate' in df_rolling.columns:
            ax.plot(df_rolling['episode'], df_rolling['success_rate'], 
                    linewidth=2, color='purple')
            ax.axhline(y=80, color='r', linestyle='--', label='80% Target')
            ax.set_ylabel('Success Rate (%)')
            ax.set_title('Success Rate Over Time (100-episode window)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Success data not available', 
                   ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        plt.show()
    
    def plot_reward_distribution(self, save_path: Optional[str] = None):
        n = len(self.df)
        early = self.df.head(n // 4)
        late = self.df.tail(n // 4)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Reward Distribution Comparison', fontsize=14, fontweight='bold')
        
        # Histogram
        ax = axes[0]
        ax.hist(early['reward_sum'], bins=30, alpha=0.6, label='Early (Q1)', color='red')
        ax.hist(late['reward_sum'], bins=30, alpha=0.6, label='Late (Q4)', color='green')
        ax.set_xlabel('Cumulative Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Box plot
        ax = axes[1]
        data_to_plot = [early['reward_sum'], late['reward_sum']]
        bp = ax.boxplot(data_to_plot, labels=['Early (Q1)', 'Late (Q4)'], patch_artist=True)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][1].set_facecolor('green')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Reward Stability')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        plt.show()
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        metrics = self.get_performance_metrics()
        
        report = []
        report.append("="*70)
        report.append(f"TRAINING ANALYSIS REPORT: {self.csv_path.stem}")
        report.append("="*70)
        report.append("")
        
        # General metrics
        report.append("OVERALL PERFORMANCE:")
        report.append(f"  Total Episodes: {metrics['total_episodes']}")
        report.append(f"  Average Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        report.append(f"  Reward Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
        report.append(f"  Average Steps/Episode: {metrics['avg_steps']:.1f} ± {metrics['std_steps']:.1f}")
        report.append("")
        
        # Accuracy metrics
        if 'avg_accuracy' in metrics:
            report.append("ACCURACY METRICS:")
            report.append(f"  Average Accuracy: {metrics['avg_accuracy']:.2f}%")
            report.append(f"  Final Accuracy (last 100 ep): {metrics['final_accuracy']:.2f}%")
            
            conv_point = self.find_convergence_point('accuracy_pct', 80.0)
            if conv_point:
                report.append(f"  Converged to 80% at episode: {conv_point}")
            else:
                report.append("  Did not converge to 80% accuracy")
            report.append("")
        
        # Success rate
        if 'success_rate' in metrics:
            report.append("SUCCESS RATE:")
            report.append(f"  Overall: {metrics['success_rate']:.2f}%")
            report.append(f"  Final (last 100 ep): {metrics['final_success_rate']:.2f}%")
            report.append("")
        
        # Quartile analysis
        report.append("LEARNING PROGRESSION (by Quartile):")
        quartiles = self.get_quartile_analysis()
        report.append(quartiles.to_string())
        report.append("")
        
        report.append("="*70)
        
        report_text = "\n".join(report)
        print(report_text)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_path}")
        
        return report_text


def analyze_episode_data(csv_path: str, 
                        generate_plots: bool = True,
                        output_dir: Optional[str] = None):
    analyzer = EpisodeAnalyzer(csv_path)
    
    # Create output directory
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(exist_ok=True, parents=True)
    else:
        out_path = Path("analysis_output")
        out_path.mkdir(exist_ok=True)
    
    # Generate report
    report_file = out_path / f"report_{Path(csv_path).stem}.txt"
    analyzer.generate_report(str(report_file))
    
    # Generate plots
    if generate_plots:
        plot_file = out_path / f"learning_curves_{Path(csv_path).stem}.png"
        analyzer.plot_learning_curves(str(plot_file))
        
        dist_file = out_path / f"reward_dist_{Path(csv_path).stem}.png"
        analyzer.plot_reward_distribution(str(dist_file))
    
    return analyzer


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_episodes.py <path_to_csv_file>")
        print("\nExample: python analyze_episodes.py Logs/training_log_20240115.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    # Run full analysis
    analyzer = analyze_episode_data(csv_file, generate_plots=True)
