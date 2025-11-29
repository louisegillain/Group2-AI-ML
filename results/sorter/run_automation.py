import os
import subprocess
import time
import yaml
import psutil

# List of training runs
configs = [ #change to the runs you want to do
    {"run_id": "Sorter_base", "reward_mode": 1, "buffer_size": 40960, "gamma": 0.99, "learning_rate": 0.0003},
    {"run_id": "Sorter_rewardmode_movingfast","reward_mode": 1, "buffer_size": 40960, "gamma": 0.99, "learning_rate": 0.0003},
    {"run_id": "Sorter_rewardmode_early","reward_mode": 2, "buffer_size": 40960, "gamma": 0.99, "learning_rate": 0.0003},
    {"run_id": "Sorter_rewardmode_still","reward_mode": 3, "buffer_size": 40960, "gamma": 0.99, "learning_rate": 0.0003},
    {"run_id": "Sorter_rewardmode_fast&early","reward_mode": 4, "buffer_size": 40960, "gamma": 0.99, "learning_rate": 0.0003},
    {"run_id": "Sorter_rewardmode_early&still","reward_mode": 5, "buffer_size": 40960, "gamma": 0.99, "learning_rate": 0.0003},
    {"run_id": "Sorter_rewardmode_fast&still","reward_mode": 6, "buffer_size": 40960, "gamma": 0.99, "learning_rate": 0.0003},
    {"run_id": "Sorter_rewardmode_fast&early&still","reward_mode": 7, "buffer_size": 40960, "gamma": 0.99, "learning_rate": 0.0003},
    {"run_id": "Sorter_buffersize_10240","reward_mode": 0, "buffer_size": 10240, "gamma": 0.99, "learning_rate": 0.0003},
    {"run_id": "Sorter_buffersize_20480","reward_mode": 0, "buffer_size": 20480, "gamma": 0.99, "learning_rate": 0.0003},
    {"run_id": "Sorter_buffersize_81920","reward_mode": 0, "buffer_size": 81920, "gamma": 0.99, "learning_rate": 0.0003},
    {"run_id": "Sorter_buffersize_163840","reward_mode": 0, "buffer_size": 163840, "gamma": 0.99, "learning_rate": 0.0003},
    {"run_id": "Sorter_gamma_050","reward_mode": 0, "buffer_size": 40960, "gamma": 0.50, "learning_rate": 0.0003},
    {"run_id": "Sorter_gamma_070","reward_mode": 0, "buffer_size": 40960, "gamma": 0.70, "learning_rate": 0.0003},
    {"run_id": "Sorter_gamma_090","reward_mode": 0, "buffer_size": 40960, "gamma": 0.90, "learning_rate": 0.0003},
    {"run_id": "Sorter_gamma_095","reward_mode": 0, "buffer_size": 40960, "gamma": 0.95, "learning_rate": 0.0003},
    {"run_id": "Sorter_learningrate_000001","reward_mode": 0, "buffer_size": 40960, "gamma": 0.99, "learning_rate": 0.00001},
    {"run_id": "Sorter_learningrate_00001","reward_mode": 0, "buffer_size": 40960, "gamma": 0.99, "learning_rate": 0.0001},
    {"run_id": "Sorter_learningrate_0001","reward_mode": 0, "buffer_size": 40960, "gamma": 0.99, "learning_rate": 0.001},
    {"run_id": "Sorter_learningrate_001","reward_mode": 0, "buffer_size": 40960, "gamma": 0.99, "learning_rate": 0.01},    
]

yaml_path = "C:/Users/louis/ml-agents/config/ppo/Sorter_curriculum.yaml" #change path to the yaml in your computer
generated_dir = "C:/Users/louis/ml-agents/config/ppo/generated" #change (this is to have a copy of the modified yaml, important otherwise it runs with original parameters, it can be whereever in your computer)
os.makedirs(generated_dir, exist_ok=True)

def monitor_training(cmd, run_id):
    start_time = time.time()
    process = subprocess.Popen(cmd)
    parent = psutil.Process(process.pid)

    def tree_rss_bytes():
        try:
            procs = [parent] + parent.children(recursive=True)
        except psutil.NoSuchProcess:
            return 0
        rss = 0
        for p in procs:
            try:
                rss += p.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return rss

    ram_samples = []
    peak_ram = 0
    min_ram = float("inf")

    try:
        while process.poll() is None:
            mem_bytes = tree_rss_bytes()
            mem_kb = mem_bytes / 1024  # or / (1024**2) for MB
            ram_samples.append(mem_kb)
            peak_ram = max(peak_ram, mem_kb)
            min_ram = min(min_ram, mem_kb)
            time.sleep(1)
    except psutil.NoSuchProcess:
        pass

    end_time = time.time()
    duration = end_time - start_time
    avg_ram = sum(ram_samples) / len(ram_samples) if ram_samples else 0
    if min_ram == float("inf"):
        min_ram = 0

    result_dir = os.path.join("results", run_id)
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "time_RAM_details.txt"), "w") as f:
        f.write(f"Wall-clock time in seconds = {duration:.5f}\n")
        f.write(f"Peak RAM usage in KB = {peak_ram:.5f}\n")
        f.write(f"Minimum RAM usage in KB = {min_ram:.5f}\n")
        f.write(f"Average RAM usage in KB = {avg_ram:.5f}\n")

    ram_log_path = os.path.join(result_dir, "ram_usage_over_time.csv")
    with open(ram_log_path, "w") as f:
        f.write("Second;RAM_KB\n")
        for i, mem in enumerate(ram_samples):
            f.write(f"{i};{mem:.50f}\n")

for cfg in configs:
    print(f"Running training: {cfg['run_id']} with reward mode: {cfg['reward_mode']}, buffer size: {cfg['buffer_size']}, gamma : {cfg['gamma']} and learning rate: {cfg['learning_rate']}")
    # change above for your specific runs (cfg['parameter'] takes the parameter value in each run)

    # Load and modify YAML
    # change here for every parameter you are changing, you basically need to put the "path" to the parameter in the yaml file in the [] (each [] is a "level" of the yaml, look at the ones written to understand)
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    config["environment_parameters"]["reward_mode"] = {
            "sampler_type": "constant",
            "sampler_parameters": {
            "value": cfg["reward_mode"]
            }  #this was kind of a exception case, yours should look like the other ones normally
    }
    config["behaviors"]["Sorter"]["hyperparameters"]["buffer_size"] = cfg["buffer_size"]
    config["behaviors"]["Sorter"]["hyperparameters"]["gamma"] = cfg["gamma"]
    config["behaviors"]["Sorter"]["hyperparameters"]["learning_rate"] = cfg["learning_rate"]

    # Save new YAML
    new_yaml_path = os.path.join(generated_dir, f"{cfg['run_id']}.yaml")
    with open(new_yaml_path, "w") as f:
        yaml.dump(config, f)
    
    train_cmd = [
        "mlagents-learn",
        new_yaml_path,
        "--run-id", cfg["run_id"],
        "--env", "C:/Users/louis/ml-agents/UnityEnvironment.exe", # change to path to the builer
        "--no-graphics"
    ]
    monitor_training(train_cmd, cfg["run_id"])
    
    time.sleep(5) #Wait before next run