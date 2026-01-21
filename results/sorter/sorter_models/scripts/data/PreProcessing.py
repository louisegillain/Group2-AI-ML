import pandas as pd
import numpy as np
from pathlib import Path


def verifyContent(run_path) : 
    if not (run_path / "configuration.yaml").exists():
        print(run_path.name + " is missing configuration yaml file.")

    if not (run_path / "ram_usage_over_time.csv").exists():
        print(run_path.name + " is missing RAM usage over time csv file.")

    if not (run_path / "time_RAM_details.txt").exists():
        print(run_path.name + " is missing the time and RAM details text file.")

    if not (run_path / "run_logs" / "timers.json").exists():
        print(run_path.name + " is missing the timers json file.")
    

def detectAnomalies(series):
    # verify anomalies with z-score
    z = (series - series.mean()) / series.std()
    return abs(z) > 3

def verifyRAM(run_path) :
    ram_df = pd.read_csv(run_path / "ram_usage_over_time.csv", sep=";")
    filtered_df = ram_df[ram_df["Second"] > 100] 
    ram_series = filtered_df["RAM_KB"] 
    
    mean = ram_series.mean() 
    std = ram_series.std() 
    
    detection = detectAnomalies(ram_series) 
    anomalies = filtered_df[detection]
    if anomalies.empty :
        print("No anomalies detected for run " + run_path.name)
    else :
        for _, row in anomalies.iterrows():
            ram_value = row["RAM_KB"]
            diff = ram_value - mean
            z = diff / std if std > 0 else float("inf")
            print(f"Anomaly detected at time {row['Second']}: RAM = {ram_value} KB, with mean = {mean}, diff = {diff}, z = {z}")

if __name__ == "__main__":
    runs_dir = Path(
        "C:\\Users\\louis\\Group2-AI-ML\\results\\sorter\\YAMLchanged_Louise\\pairwise tests\\layer & hidden"
    )
    for run in runs_dir.iterdir():
        print(run.name)
        run_path = run
        verifyContent(run_path)
        verifyRAM(run_path)
