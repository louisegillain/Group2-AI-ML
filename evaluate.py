import time
import numpy as np
import pandas as pd
import onnxruntime as ort
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

ENV_BUILD_PATH = "Project/Builds/Soccer/UnityEnvironment.exe"
BLUE_MODEL_PATH = "results\S\SoccerTwos.onnx"
PURPLE_MODEL_PATH = "results\S\SoccerTwos.onnx"

BLUE = "BlueTeam?team=0"
PURPLE = "PurpleTeam?team=1"

NUMBER_MATCH = 100
results = []
matches_played = 0

# Load ONNX models
try:
    blue_session = ort.InferenceSession(BLUE_MODEL_PATH, providers=['CPUExecutionProvider'])
    purple_session = ort.InferenceSession(PURPLE_MODEL_PATH, providers=['CPUExecutionProvider'])
    print("Models loaded")
except Exception as e:
    print("Error loading ONNX models:", e)
    exit()

# Launch environment
env = UnityEnvironment(file_name=ENV_BUILD_PATH, worker_id=1, no_graphics=False)
env.reset()
print("Environment launched")

def get_actions(session, decision_steps):
    obs_0 = decision_steps.obs[0].astype(np.float32)
    obs_1 = decision_steps.obs[1].astype(np.float32)

    # Handle action masks
    if isinstance(decision_steps.action_mask, list):
        action_masks = np.concatenate(decision_steps.action_mask, axis=1).astype(np.float32)
    else:
        action_masks = decision_steps.action_mask.astype(np.float32)

    input_feed = {
        "obs_0": obs_0,
        "obs_1": obs_1,
        "action_masks": action_masks
    }

    raw_actions = session.run(["discrete_actions"], input_feed)[0]
    return ActionTuple(discrete=raw_actions)

game_over = False
starting_time = time.time()

try:
    while matches_played < NUMBER_MATCH:
        

        # BLUE TEAM
        decision_steps_blue, terminal_steps_blue = env.get_steps(BLUE)
        
        # PURPLE TEAM
        decision_steps_purple, terminal_steps_purple = env.get_steps(PURPLE)

        if len(terminal_steps_blue) > 0:
                    matches_played += 1
                    
                    reward = terminal_steps_blue.reward[0]
                    winner = "Draw"
                    if reward > 0:
                        winner = "Blue"
                    elif reward < 0:
                        winner = "Purple"
                    
                    print(f"Match {matches_played}/{NUMBER_MATCH} complete. Winner: {winner}")
                    
                    results.append({
                        "match_id": matches_played,
                        "blue_model": BLUE_MODEL_PATH,
                        "purple_model": PURPLE_MODEL_PATH,
                        "winner": winner
                    })

        if len(decision_steps_blue) > 0:
            action_blue = get_actions(blue_session, decision_steps_blue)
            env.set_actions(BLUE, action_blue)

        if len(decision_steps_purple) > 0:
            action_purple = get_actions(purple_session, decision_steps_purple)
            env.set_actions(PURPLE, action_purple)

        env.step()

except KeyboardInterrupt:
    print("Run interrupted by user.")

finally:
    env.close()
    ending_time = time.time()
    print("Evaluation finished. Environment closed. \n running time =" + str(ending_time - starting_time))

    df = pd.DataFrame(data=results)
    df.to_csv("matches_results.csv", index=False)
    print("data logged.")
