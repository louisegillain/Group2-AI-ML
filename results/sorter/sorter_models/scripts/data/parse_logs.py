import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("Installing tensorboard for log parsing...")
    os.system("pip install tensorboard")
    from tensorboard.backend.event_processing import event_accumulator


class MLAgentsLogParser:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.data: Dict[str, pd.DataFrame] = {}
        
    def find_event_files(self) -> List[Path]:
        event_files = list(self.log_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            raise FileNotFoundError(f"No event files found in {self.log_dir}")
        return event_files
    
    def parse_event_file(self, event_file: Path) -> Dict[str, pd.DataFrame]:
        ea = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={
                event_accumulator.SCALARS: 0,  # Load all scalars
            }
        )
        ea.Reload()
        
        metrics = {}
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            df = pd.DataFrame([
                {'step': e.step, 'value': e.value, 'wall_time': e.wall_time}
                for e in events
            ])
            metrics[tag] = df
            
        return metrics
    
    def parse_all_logs(self) -> Dict[str, pd.DataFrame]:
        event_files = self.find_event_files()
        print(f"Found {len(event_files)} event file(s)")
        
        all_metrics = {}
        
        for event_file in event_files:
            print(f"Parsing: {event_file.name}")
            metrics = self.parse_event_file(event_file)
            
            for tag, df in metrics.items():
                if tag in all_metrics:
                    all_metrics[tag] = pd.concat([all_metrics[tag], df], ignore_index=True)
                else:
                    all_metrics[tag] = df
        
        for tag in all_metrics:
            all_metrics[tag] = all_metrics[tag].sort_values('step').drop_duplicates('step')
        
        self.data = all_metrics
        return all_metrics
    
    def get_metric(self, metric_name: str) -> Optional[pd.DataFrame]:
        """Get a specific metric DataFrame"""
        return self.data.get(metric_name)
    
    def get_cumulative_reward(self) -> pd.DataFrame:
        """Get cumulative reward metric"""
        return self.get_metric('Environment/Cumulative Reward')
    
    def get_episode_length(self) -> pd.DataFrame:
        """Get episode length metric"""
        return self.get_metric('Environment/Episode Length')
    
    def get_policy_loss(self) -> pd.DataFrame:
        """Get policy loss metric"""
        return self.get_metric('Losses/Policy Loss')
    
    def get_value_loss(self) -> pd.DataFrame:
        """Get value loss metric"""
        return self.get_metric('Losses/Value Loss')
    
    def list_available_metrics(self) -> List[str]:
        """List all available metrics in the parsed logs"""
        return sorted(self.data.keys())
    
    def export_to_csv(self, output_dir: str = "parsed_logs"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for metric_name, df in self.data.items():
            # Create safe filename
            safe_name = metric_name.replace('/', '_').replace(' ', '_')
            csv_path = output_path / f"{safe_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Exported: {csv_path}")
    
    def get_summary_stats(self) -> pd.DataFrame:
        stats = []
        
        key_metrics = [
            'Environment/Cumulative Reward',
            'Environment/Episode Length',
            'Losses/Policy Loss',
            'Losses/Value Loss'
        ]
        
        for metric in key_metrics:
            df = self.get_metric(metric)
            if df is not None and not df.empty:
                stats.append({
                    'metric': metric,
                    'min': df['value'].min(),
                    'max': df['value'].max(),
                    'mean': df['value'].mean(),
                    'std': df['value'].std(),
                    'final': df['value'].iloc[-1],
                    'steps': len(df)
                })
        
        return pd.DataFrame(stats)


def parse_logs(log_dir: str, export_csv: bool = True) -> MLAgentsLogParser:
    parser = MLAgentsLogParser(log_dir)
    parser.parse_all_logs()
    
    print(f"\nAvailable metrics: {len(parser.list_available_metrics())}")
    print("\nSummary Statistics:")
    print(parser.get_summary_stats().to_string(index=False))
    
    if export_csv:
        parser.export_to_csv()
    
    return parser


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parse_logs.py <path_to_results_directory>")
        print("\nExample: python parse_logs.py results/SorterGame/PPO")
        sys.exit(1)
    
    log_directory = sys.argv[1]
    
    if not os.path.exists(log_directory):
        print(f"Error: Directory not found: {log_directory}")
        sys.exit(1)
    
    # Parse logs and export to CSV
    parser = parse_logs(log_directory, export_csv=True)
    
    # Display available metrics
    print("\n" + "="*60)
    print("All Available Metrics:")
    print("="*60)
    for metric in parser.list_available_metrics():
        print(f"  - {metric}")