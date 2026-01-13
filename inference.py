import argparse
import os
import time
import yaml
import numpy as np
import pandas as pd
import onnxruntime as ort
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


ENV_BUILD_PATH = None
BLUE_MODEL_PATH = None
PURPLE_MODEL_PATH = None
BLUE = None
PURPLE = None
NUMBER_MATCH = None

results = []
matches_played = 0

def main():
    global ENV_BUILD_PATH, NUMBER_MATCH
    global BLUE_MODEL_PATH, PURPLE_MODEL_PATH
    global BLUE, PURPLE

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--number_match", type=int, help="Override number of matches")
    args = parser.parse_args()

    config = args.config
    if args.number_match:
        NUMBER_MATCH = args.number_match


    if not os.path.exists(config):
        raise FileNotFoundError(f"Config file {config} does not exist")


    with open(config, 'r') as c:
        cfg = yaml.safe_load(c)

    ENV_BUILD_PATH = cfg["environment"]["build_path"]
    if not os.path.exists(ENV_BUILD_PATH):
        raise FileNotFoundError(f"Environment {ENV_BUILD_PATH} does not exist")
    
    NUMBER_MATCH = cfg["environment"]["number_match"]

    BLUE_MODEL_PATH = cfg["models"]["blue_team_path"]
    PURPLE_MODEL_PATH = cfg["models"]["purple_team_path"]

    if not os.path.exists(PURPLE_MODEL_PATH):
        raise FileNotFoundError(f"Model {PURPLE_MODEL_PATH} does not exist")
    
    if not os.path.exists(BLUE_MODEL_PATH):
        raise FileNotFoundError(f"Model {BLUE_MODEL_PATH} does not exist")

    BLUE = cfg["teams"]["blue_id"]
    PURPLE = cfg["teams"]["purple_id"]

    print("Configuration loaded:")
    print(cfg)



def get_actions(session, decision_steps):
    """Generates actions from the loaded ONNX model."""
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

    print(input_feed)

    # Run inference to get discrete actions
    raw_actions = session.run(["discrete_actions"], input_feed)[0]
    return ActionTuple(discrete=raw_actions), raw_actions


def recordStepData(team, decision_steps, actions):
    """
    Extracts agent and ball data for the decision step and appends to the results list.
    
    :param team: The team name ("Blue" or "Purple").
    :param decision_steps: The DecisionSteps object for the current team.
    :param actions: The discrete actions returned by the model for this step.
    """
    global global_current_step, blue_score, purple_score

    for i in range(len(decision_steps.agent_id)):
        agent_id = decision_steps.agent_id[i]

        position = "Goalie" if agent_id % 2 == 0 else "Striker"
        
        agent_obs = decision_steps.obs[1][i] 
        
        agent_pos_x, agent_pos_y, agent_pos_z = agent_obs[0], agent_obs[1], agent_obs[2]
        agent_vel_x, agent_vel_y, agent_vel_z = agent_obs[3], agent_obs[4], agent_obs[5]
        
        ball_x, ball_y, ball_z = np.nan, np.nan, np.nan # Global ball pos
        ball_x_vel, ball_y_vel, ball_z_vel = np.nan, np.nan, np.nan # Global ball vel

        goal_distance = agent_obs[6]
        
        a = actions[i]
        move_forward = 1 if a[0] == 1 else 0
        move_back    = 1 if a[0] == 2 else 0
        move_right   = 1 if a[1] == 1 else 0
        move_left    = 1 if a[1] == 2 else 0
        rotate_left  = 1 if a[2] == 1 else 0
        rotate_right = 1 if a[2] == 2 else 0
        

        current_score = f"{blue_score}-{purple_score}"

        results.append({
            "match_id": matches_played,
            "frame": global_current_step,
            "team": team,
            "position": position,
            "agent_x": agent_pos_x,
            "agent_y": agent_pos_y,
            "agent_z": agent_pos_z,
            "agent_x_vel": agent_vel_x,
            "agent_y_vel": agent_vel_y,
            "agent_z_vel": agent_vel_z,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "ball_z": ball_z,
            "ball_x_vel": ball_x_vel,
            "ball_y_vel": ball_y_vel,
            "ball_z_vel": ball_z_vel,
            "goal_distance": goal_distance,
            "game_score": current_score,
            "move_forward": move_forward,
            "move_back": move_back,
            "move_right": move_right,
            "move_left": move_left,
            "rotate_left": rotate_left,
            "rotate_right": rotate_right,
            "reward": decision_steps.reward[i],
        })


if __name__ == "__main__":
    main()

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

global_current_step = 0
blue_score = 0
purple_score = 0
starting_time = time.time()

try:
    while matches_played < NUMBER_MATCH:
        
        # Get steps for both teams
        decision_steps_blue, terminal_steps_blue = env.get_steps(BLUE)
        decision_steps_purple, terminal_steps_purple = env.get_steps(PURPLE)

        if len(terminal_steps_blue) > 0:
            matches_played += 1
            global_current_step = 0
            
            reward = terminal_steps_blue.reward[0] 
            winner = "Draw"
            
            # The game reward
            if reward > 0:
                winner = "Blue"
                blue_score += 1
            elif reward < 0:
                winner = "Purple"
                purple_score += 1
            
            print(f"Match {matches_played}/{NUMBER_MATCH} complete. Winner: {winner}. Score: {blue_score}-{purple_score}")
            
            # Record final match outcome summary
            results.append({
                "match_id": matches_played,
                "blue_model": BLUE_MODEL_PATH,
                "purple_model": PURPLE_MODEL_PATH,
                "winner": winner
            })
            
        if len(decision_steps_blue) > 0:
            action_tuple_blue, raw_actions_blue = get_actions(blue_session, decision_steps_blue)
            env.set_actions(BLUE, action_tuple_blue)
            recordStepData("Blue", decision_steps_blue, raw_actions_blue)

        if len(decision_steps_purple) > 0:
            action_tuple_purple, raw_actions_purple = get_actions(purple_session, decision_steps_purple)
            env.set_actions(PURPLE, action_tuple_purple)
            recordStepData("Purple", decision_steps_purple, raw_actions_purple)

        global_current_step += 1
        env.step()

except KeyboardInterrupt:
    print("Run interrupted by user.")

finally:
    env.close()
    ending_time = time.time()
    print(f"Evaluation finished. Environment closed. \nRunning time = {ending_time - starting_time:.2f} seconds")

    # The final data frame might contain two types of records: step data and match results.
    df = pd.DataFrame(data=results)
    
    # Save step data
    step_data_df = df[df['frame'].notna()]
    step_data_df.to_csv("step_data_results.csv", index=False)
    print("Step data logged to step_data_results.csv")
    
    # Save match results
    match_results_df = df[df['winner'].notna()]
    match_results_df.to_csv("matches_summary.csv", index=False)
    print("Match summary logged to matches_summary.csv")