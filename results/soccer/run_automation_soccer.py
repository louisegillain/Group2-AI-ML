import os
import subprocess
import time
import yaml
import psutil
import csv
import glob
from datetime import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# change these to your specific paths
yaml_path = "C:/Users/alext/PycharmProjects/Group2-AI-ML/results/soccer/config/poca/SoccerTwos.yaml"
generated_dir = "C:/Users/alext/PycharmProjects/Group2-AI-ML/results/soccer/config/ppo/generated"
env_path = "C:/Users/alext/PycharmProjects/Group2-AI-ML/Project/Build/UnityEnvironment.exe"
results_folder = "C:/Users/alext/PycharmProjects/Group2-AI-ML/results/soccer/Soccer_Training_Results"
ml_agents_results_dir = "C:/Users/alext/PycharmProjects/Group2-AI-ML/results/soccer/results"

batch_sizes = [512, 1024, 2048]
hidden_units_list = [128, 256, 512]
layers_list = [1, 2, 3]

os.makedirs(generated_dir, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

# set this to the name of your existing Master CSV file
# if left as None, it will create a brand new file
RESUME_FILE = "soccer_training_master_20260114_161701.csv"

if RESUME_FILE:
    master_csv = os.path.join(results_folder, RESUME_FILE)
    session_ts = RESUME_FILE.split('_')[-1].replace('.csv', '')
else:
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_csv = os.path.join(results_folder, f"soccer_training_master_{session_ts}.csv")

time_csv = os.path.join(results_folder, f"time_only_{session_ts}.csv")
ram_csv = os.path.join(results_folder, f"ram_only_{session_ts}.csv")


def get_completed_combinations(filepath):
    #reads the CSV file
    completed = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    completed.add((int(row['batch_size']), int(row['hidden_units']), int(row['num_layers'])))
        except Exception as e:
            print(f"Warning: Could not read resume file: {e}")
    return completed


def log_to_csv(filepath, headers, data):
    # longs data to CSV
    file_exists = os.path.isfile(filepath)
    for attempt in range(10):  # Try for 50 seconds total
        try:
            with open(filepath, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists: writer.writeheader()
                writer.writerow(data)
            return True
        except PermissionError:
            print(f"!!! error: {filepath} is open in Excel. Please close it. Retrying in 5s...")
            time.sleep(5)
    return False


def get_latest_reward(run_id):
    try:
        path = os.path.join(ml_agents_results_dir, run_id, "**", "*.tfevents.*")
        event_files = glob.glob(path, recursive=True)
        if not event_files: return 0.0
        latest_file = max(event_files, key=os.path.getmtime)
        event_acc = EventAccumulator(latest_file)
        event_acc.Reload()
        tags = event_acc.Tags()['scalars']
        possible_tags = ['Environment/Cumulative Reward', 'Environment/Group Cumulative Reward',
                         'Policy/Cumulative Reward']
        for tag in possible_tags:
            if tag in tags:
                rewards = event_acc.Scalars(tag)
                if rewards: return rewards[-1].value
    except Exception:
        return 0.0
    return 0.0


def monitor_training(cmd, run_id, threshold):
    start_time = time.time()
    process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    ram_samples, cpu_samples, last_reward = [], [], 0.0
    try:
        parent = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                children = parent.children(recursive=True)
                rss = sum(p.memory_info().rss for p in [parent] + children if p.is_running())
                ram_samples.append(rss / 1024)
                cpu_samples.append(psutil.cpu_percent(interval=None))
                current_reward = get_latest_reward(run_id)
                if current_reward != 0: last_reward = current_reward
                print(f"   [Monitoring] {run_id} | Reward: {last_reward:.2f} / {threshold}")
                if last_reward >= threshold and last_reward != 0:
                    for child in children: child.kill()
                    parent.kill()
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(5)
        duration = time.time() - start_time
        return duration, max(ram_samples or [0]), (sum(ram_samples) / len(ram_samples) if ram_samples else 0), max(
            cpu_samples or [0]), last_reward
    except KeyboardInterrupt:
        process.kill()
        raise


print(f"--- Session: {session_ts} ---")
finished_combos = get_completed_combinations(master_csv)
run_num = len(finished_combos) + 1

try:
    while True:
        for b in batch_sizes:
            for h in hidden_units_list:
                for l in layers_list:
                    # CHECK IF ALREADY DONE
                    if (b, h, l) in finished_combos:
                        continue

                    run_ts = datetime.now().strftime("%H%M%S")
                    full_run_id = f"Soccer_B{b}_H{h}_L{l}_{run_ts}"
                    print(f"\n>>> Run #{run_num}: {full_run_id}")

                    # YAML Modification
                    with open(yaml_path, "r") as f:
                        yaml_data = yaml.safe_load(f)

                    behavior_name = "SoccerTwos"
                    yaml_data["behaviors"][behavior_name]["max_steps"] = 3000000
                    yaml_data["behaviors"][behavior_name]["summary_freq"] = 10000
                    yaml_data["behaviors"][behavior_name]["hyperparameters"]["batch_size"] = b
                    yaml_data["behaviors"][behavior_name]["network_settings"]["hidden_units"] = h
                    yaml_data["behaviors"][behavior_name]["network_settings"]["num_layers"] = l

                    new_yaml_path = os.path.join(generated_dir, f"{full_run_id}.yaml")
                    with open(new_yaml_path, "w") as f:
                        yaml.dump(yaml_data, f)

                    train_cmd = ["mlagents-learn", new_yaml_path, "--run-id", full_run_id, "--env", env_path,
                                 "--no-graphics"]
                    duration, peak_ram, avg_ram, peak_cpu, final_reward = monitor_training(train_cmd, full_run_id, -1.1)


                    headers_master = ["run_id", "batch_size", "hidden_units", "num_layers", "duration_sec",
                                      "peak_ram_kb", "avg_ram_kb"]
                    data_payload = {
                        "run_id": full_run_id, "batch_size": b, "hidden_units": h, "num_layers": l,
                        "duration_sec": f"{duration:.2f}", "peak_ram_kb": f"{peak_ram:.2f}",
                        "avg_ram_kb": f"{avg_ram:.2f}"
                    }

                    if log_to_csv(master_csv, headers_master, data_payload):
                        log_to_csv(time_csv, ["run_id", "batch_size", "hidden_units", "num_layers", "duration_sec",
                                              "final_reward"],
                                   {"run_id": full_run_id, "batch_size": b, "hidden_units": h, "num_layers": l,
                                    "duration_sec": f"{duration:.2f}", "final_reward": f"{final_reward:.2f}"})
                        log_to_csv(ram_csv,
                                   ["run_id", "batch_size", "hidden_units", "num_layers", "peak_ram_kb", "avg_ram_kb"],
                                   {"run_id": full_run_id, "batch_size": b, "hidden_units": h, "num_layers": l,
                                    "peak_ram_kb": f"{peak_ram:.2f}", "avg_ram_kb": f"{avg_ram:.2f}"})

                        finished_combos.add((b, h, l))
                        print(f"Finished Run #{run_num}. Data saved.")
                        run_num += 1

                    time.sleep(10)
except KeyboardInterrupt:
    print("\nStopped.")
