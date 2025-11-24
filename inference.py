import time
import numpy as np
import pandas as pd
import onnxruntime as ort
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

ENV_BUILD_PATH = "Project/Builds/Soccer/UnityEnvironment.exe"
BLUE_MODEL_PATH = "C:/Users/franc/ml-agents/Project/Assets/ML-Agents/Examples/Soccer/TFModels/SoccerTwos.onnx"
PURPLE_MODEL_PATH = "C:/Users/franc/ml-agents/Project/Assets/ML-Agents/Examples/Soccer/TFModels/SoccerTwos.onnx"

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

global_current_step = 0
blue_score = 0
purple_score = 0


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

        goal_distance = agent_obs[6] # relative x of ball and agent
        
        discrete_action = actions[i][0]

        move_forward = 1 if discrete_action == 1 else 0
        move_back = 1 if discrete_action == 2 else 0
        rotate_left = 1 if discrete_action == 3 else 0
        rotate_right = 1 if discrete_action == 4 else 0
        move_right = 1 if discrete_action == 5 else 0
        move_left = 1 if discrete_action == 6 else 0
        

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