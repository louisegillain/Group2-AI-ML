import os
import subprocess
import time
import yaml
import psutil
from pathlib import Path

# List of training 

# Pattern to follow to write the runs :
# {"run_id": "*name of your run*", *list of hyperparameters*},
# The list of hyperparameters should be presented as follows :
# For each hyperparameters -> "*name of hyperparameter*": *value*
# If you have more than 1 hyperparameter, separate them with a comma

# A few examples are provided just under.
configs = [ #change to the runs you want to do
    {"run_id": "Sorter_batchL&bufferL", "buffer_size": 10240,"batch_size": 128, "hidden_units": 128, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_batchL&bufferH", "buffer_size": 163840,"batch_size": 128, "hidden_units": 128, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_batchM&bufferM", "buffer_size": 40960,"batch_size": 512, "hidden_units": 128, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_batchH&bufferL", "buffer_size": 10240,"batch_size": 2048, "hidden_units": 128, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_batchH&bufferH", "buffer_size": 163840,"batch_size": 2048, "hidden_units": 128, "num_layers": 2, "num_envs" : 1},

    {"run_id": "Sorter_bufferL&hiddenL", "buffer_size": 10240,"batch_size": 512, "hidden_units": 64, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_bufferL&hiddenH", "buffer_size": 10240,"batch_size": 512, "hidden_units": 256, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_bufferM&hiddenM", "buffer_size": 40960,"batch_size": 512, "hidden_units": 128, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_bufferH&hiddenL", "buffer_size": 163840,"batch_size": 512, "hidden_units": 64, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_bufferH&hiddenH", "buffer_size": 163840,"batch_size": 512, "hidden_units": 256, "num_layers": 2, "num_envs" : 1},

    {"run_id": "Sorter_hiddenunits_32", "buffer_size": 40960,"batch_size": 512, "hidden_units": 32, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_hiddenunits_64", "buffer_size": 40960,"batch_size": 512, "hidden_units": 64, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_hiddenunits_96", "buffer_size": 40960,"batch_size": 512, "hidden_units": 96, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_hiddenunits_160", "buffer_size": 40960,"batch_size": 512, "hidden_units": 160, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_hiddenunits_192", "buffer_size": 40960,"batch_size": 512, "hidden_units": 192, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_hiddenunits_224", "buffer_size": 40960,"batch_size": 512, "hidden_units": 224, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_hiddenunits_256", "buffer_size": 40960,"batch_size": 512, "hidden_units": 256, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_hiddenunits_320", "buffer_size": 40960,"batch_size": 512, "hidden_units": 320, "num_layers": 2, "num_envs" : 1},
    {"run_id": "Sorter_hiddenunits_384", "buffer_size": 40960,"batch_size": 512, "hidden_units": 384, "num_layers": 2, "num_envs" : 1},

    {"run_id": "Sorter_numlayer_1", "buffer_size": 40960,"batch_size": 512, "hidden_units": 128, "num_layers": 1, "num_envs" : 1},
    {"run_id": "Sorter_numlayer_3", "buffer_size": 40960,"batch_size": 512, "hidden_units": 128, "num_layers": 3, "num_envs" : 1},
    {"run_id": "Sorter_numlayer_4", "buffer_size": 40960,"batch_size": 512, "hidden_units": 128, "num_layers": 4, "num_envs" : 1},
    {"run_id": "Sorter_numlayer_5", "buffer_size": 40960,"batch_size": 512, "hidden_units": 128, "num_layers": 5, "num_envs" : 1},
    {"run_id": "Sorter_numlayer_6", "buffer_size": 40960,"batch_size": 512, "hidden_units": 128, "num_layers": 6, "num_envs" : 1},
    {"run_id": "Sorter_numlayer_7", "buffer_size": 40960,"batch_size": 512, "hidden_units": 128, "num_layers": 7, "num_envs" : 1},
    {"run_id": "Sorter_numlayer_8", "buffer_size": 40960,"batch_size": 512, "hidden_units": 128, "num_layers": 8, "num_envs" : 1},
    {"run_id": "Sorter_numlayer_9", "buffer_size": 40960,"batch_size": 512, "hidden_units": 128, "num_layers": 9, "num_envs" : 1},
    {"run_id": "Sorter_numlayer_10", "buffer_size": 40960,"batch_size": 512, "hidden_units": 128, "num_layers": 10, "num_envs" : 1},

    {"run_id": "Sorter_envs_4","buffer_size": 40960,"batch_size": 512, "hidden_units": 128, "num_layers": 2, "num_envs" : 4},
    {"run_id": "Sorter_envs_8","buffer_size": 40960,"batch_size": 512, "hidden_units": 128, "num_layers": 2, "num_envs" : 8},
]

base_dir = Path(__file__).resolve().parents[3]
yaml_path = base_dir / "Sorter_curriculum.yaml"
generated_dir = base_dir / "generated yaml" # this is to have a copy of the modified yaml, important otherwise it runs with original parameters, it can be whereever in your computer
os.makedirs(generated_dir, exist_ok=True)

# Extract the RAM and time details, puts them in a txt file and create a csv with RAM usage overtime
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


# Runs the scripts for each run
for cfg in configs:
    print(f"Running training: {cfg['run_id']}")

    # Load and modify YAML
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    config["behaviors"]["Sorter"]["hyperparameters"]["buffer_size"] = cfg["buffer_size"]
    config["behaviors"]["Sorter"]["hyperparameters"]["batch_size"] = cfg["batch_size"]
    config["behaviors"]["Sorter"]["network_settings"]["hidden_units"] = cfg["hidden_units"]
    config["behaviors"]["Sorter"]["network_settings"]["num_layers"] = cfg["num_layers"]

    # Save new YAML
    new_yaml_path = os.path.join(generated_dir, f"{cfg['run_id']}.yaml")
    with open(new_yaml_path, "w") as f:
        yaml.dump(config, f)
    
    builder_path = base_dir / "UnityEnvironment.exe"
    results_dir = base_dir / "YAMLchanged_Louise"
    train_cmd = [
        "mlagents-learn",
        new_yaml_path,
        "--run-id", cfg["run_id"],
        "--results-dir", results_dir, 
        "--env", builder_path,
        "--no-graphics"
        # "--num-envs", str(cfg["num_envs"]) 
    ]
    monitor_training(train_cmd, cfg["run_id"])
    
    time.sleep(5) #Wait before next run