import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class ConfigComparator:
    
    def __init__(self):
        self.configs: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}
    
    def add_config(self, name: str, csv_path: str, metadata: Optional[Dict] = None):
        df = pd.read_csv(csv_path)
        self.configs[name] = df
        self.metadata[name] = metadata or {}
        print(f"Added config '{name}': {len(df)} episodes")
    
    def compute_summary_table(self) -> pd.DataFrame:
        summaries = []
        
        for name, df in self.configs.items():
            final_df = df.tail(100)
            
            summary = {
                'Config': name,
                'Episodes': len(df),
                'Avg Reward': df['reward_sum'].mean(),
                'Final Reward': final_df['reward_sum'].mean(),
                'Reward Std': df['reward_sum'].std(),
                'Avg Steps': df['steps'].mean(),
                'Final Steps': final_df['steps'].mean()
            }
            
            if 'accuracy_pct' in df.columns:
                summary['Avg Accuracy'] = df['accuracy_pct'].mean()
                summary['Final Accuracy'] = final_df['accuracy_pct'].mean()
            
            if 'success' in df.columns:
                summary['Success Rate'] = df['success'].mean() * 100
                summary['Final Success'] = final_df['success'].mean() * 100
            
            if 'accuracy_pct' in df.columns:
                conv = self._find_convergence(df, 'accuracy_pct', 80.0)
                summary['Conv. Episode (80%)'] = conv if conv else 'N/A'
            
            if self.metadata[name]:
                for key, val in self.metadata[name].items():
                    summary[key] = val
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def _find_convergence(self, df: pd.DataFrame, 
                         metric: str, threshold: float, 
                         window: int = 100) -> Optional[int]:
        """Find convergence point for a metric"""
        if metric not in df.columns:
            return None
        rolling = df[metric].rolling(window, min_periods=window).mean()
        converged = rolling >= threshold
        if converged.any():
            return int(df.loc[converged.idxmax(), 'episode'])
        return None
    
    def perform_statistical_test(self, 
                                 metric: str = 'reward_sum',
                                 window: int = 100) -> pd.DataFrame:

        config_names = list(self.configs.keys())
        n = len(config_names)
        
        results = []
        
        for i in range(n):
            for j in range(i + 1, n):
                name1, name2 = config_names[i], config_names[j]
                data1 = self.configs[name1][metric].tail(window)
                data2 = self.configs[name2][metric].tail(window)
                
                t_stat, p_value = stats.ttest_ind(data1, data2)
                
                pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
                cohens_d = (data1.mean() - data2.mean()) / pooled_std
                
                results.append({
                    'Comparison': f"{name1} vs {name2}",
                    'Mean Diff': data1.mean() - data2.mean(),
                    't-statistic': t_stat,
                    'p-value': p_value,
                    "Cohen's d": cohens_d,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
        
        return pd.DataFrame(results)
    
    def plot_learning_curves_comparison(self, 
                                       metric: str = 'reward_sum',
                                       window: int = 100,
                                       save_path: Optional[str] = None):
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = sns.color_palette("husl", len(self.configs))
        
        for (name, df), color in zip(self.configs.items(), colors):
            if metric not in df.columns:
                continue
            
            rolling_mean = df[metric].rolling(window, min_periods=1).mean()
            rolling_std = df[metric].rolling(window, min_periods=1).std()
            
            ax.plot(df['episode'], rolling_mean, label=name, 
                   linewidth=2.5, color=color)
            ax.fill_between(df['episode'],
                           rolling_mean - rolling_std,
                           rolling_mean + rolling_std,
                           alpha=0.15, color=color)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison (Rolling Mean, window={window})', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        plt.show()
    
    def plot_final_performance_comparison(self, 
                                         metrics: List[str] = None,
                                         window: int = 100,
                                         save_path: Optional[str] = None):
        
        if metrics is None:
            metrics = ['reward_sum', 'accuracy_pct', 'steps']
        
        available_metrics = []
        for metric in metrics:
            if any(metric in df.columns for df in self.configs.values()):
                available_metrics.append(metric)
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, available_metrics):
            data = []
            names = []
            
            for name, df in self.configs.items():
                if metric in df.columns:
                    final_value = df[metric].tail(window).mean()
                    data.append(final_value)
                    names.append(name)
            
            colors = sns.color_palette("husl", len(data))
            bars = ax.bar(names, data, color=colors, alpha=0.7, edgecolor='black')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'Final {metric.replace("_", " ").title()}\n(Last {window} episodes)', 
                        fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        plt.show()
    
    def plot_convergence_comparison(self, save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        conv_data = []
        
        for name, df in self.configs.items():
            if 'accuracy_pct' in df.columns:
                for threshold in [50, 60, 70, 80, 90]:
                    conv = self._find_convergence(df, 'accuracy_pct', threshold)
                    if conv:
                        conv_data.append({
                            'Config': name,
                            'Threshold': f'{threshold}%',
                            'Episode': conv
                        })
        
        if not conv_data:
            print("No convergence data available")
            return
        
        conv_df = pd.DataFrame(conv_data)
        
        pivot = conv_df.pivot(index='Threshold', columns='Config', values='Episode')
        pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
        
        ax.set_xlabel('Accuracy Threshold', fontsize=12)
        ax.set_ylabel('Episode Number', fontsize=12)
        ax.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
        ax.legend(title='Config', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        plt.show()
    
    def generate_comparison_report(self, output_path: Optional[str] = None) -> str:
        report = []
        report.append("="*80)
        report.append("CONFIGURATION COMPARISON REPORT")
        report.append("="*80)
        report.append("")
        
        report.append("PERFORMANCE SUMMARY:")
        summary = self.compute_summary_table()
        report.append(summary.to_string(index=False))
        report.append("")
        
        if len(self.configs) >= 2:
            report.append("STATISTICAL SIGNIFICANCE TESTS (Reward):")
            stats_df = self.perform_statistical_test('reward_sum')
            report.append(stats_df.to_string(index=False))
            report.append("")
            report.append("Note: p < 0.05 indicates statistically significant difference")
            report.append("")
        
        report.append("RECOMMENDATIONS:")
        if 'Final Reward' in summary.columns:
            best_idx = summary['Final Reward'].idxmax()
            best_config = summary.loc[best_idx, 'Config']
            best_reward = summary.loc[best_idx, 'Final Reward']
            report.append(f"  Best Overall Performance: {best_config} (Final Reward: {best_reward:.2f})")
        
        if 'Conv. Episode (80%)' in summary.columns:
            conv_vals = summary['Conv. Episode (80%)'].replace('N/A', np.nan).dropna()
            if not conv_vals.empty:
                fastest_idx = conv_vals.astype(int).idxmin()
                fastest_config = summary.loc[fastest_idx, 'Config']
                fastest_ep = summary.loc[fastest_idx, 'Conv. Episode (80%)']
                report.append(f"  Fastest Convergence: {fastest_config} (Episode {fastest_ep})")
        
        report.append("")
        report.append("="*80)
        
        report_text = "\n".join(report)
        print(report_text)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_path}")
        
        return report_text


def compare_training_runs(config_files: Dict[str, str],
                         metadata: Optional[Dict[str, Dict]] = None,
                         output_dir: str = "comparison_output"):
    comparator = ConfigComparator()
    
    for name, csv_path in config_files.items():
        meta = metadata.get(name) if metadata else None
        comparator.add_config(name, csv_path, meta)
    
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    report_file = out_path / "comparison_report.txt"
    comparator.generate_comparison_report(str(report_file))
    
    comparator.plot_learning_curves_comparison(
        save_path=str(out_path / "learning_curves_comparison.png")
    )
    
    comparator.plot_final_performance_comparison(
        save_path=str(out_path / "final_performance_comparison.png")
    )
    
    comparator.plot_convergence_comparison(
        save_path=str(out_path / "convergence_comparison.png")
    )
    
    return comparator


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python compare_configs.py <name1:path1> <name2:path2> [name3:path3] ...")
        print("\nExample: python compare_configs.py baseline:logs/base.csv shaped:logs/shaped.csv")
        sys.exit(1)
    
    # Parse arguments
    config_files = {}
    for arg in sys.argv[1:]:
        if ':' not in arg:
            print(f"Error: Invalid format '{arg}'. Use name:path")
            sys.exit(1)
        name, path = arg.split(':', 1)
        config_files[name] = path
    
    # Run comparison
    comparator = compare_training_runs(config_files)