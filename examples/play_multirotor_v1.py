"""
Use terminal to run this code.

python play_multirotor.py
"""

import gym
import numpy as np
import gym_multirotor
from control_algo import DDPGAgent


def main(env):
    render = True
    # ----- Environment Info ------------------------
    obs_dimensions = env.observation_space.shape[0]
    print("Observation dimensions:", obs_dimensions)

    action_dimensions = env.action_space.shape[0]
    print("Action dimensions:", action_dimensions)

    min_action = env.action_space.low
    print("Min. action:", min_action)

    max_action = env.action_space.high
    print("Max. action:", max_action)

    print("Actuator_control:", type(env.model.actuator_ctrlrange))
    print("actuator_forcerange:", env.model.actuator_forcerange)
    print("actuator_forcelimited:", env.model.actuator_forcelimited)
    print("actuator_ctrllimited:", env.model.actuator_ctrllimited)
    # --------------------------------------------

    ob = env.reset()
    done = False
    if render:
        env.render()
    
    agent = DDPGAgent(obs_dimensions, action_dimensions, max_action)
    num_episodes = 1000
    for episode in range(num_episodes):
    	state = env.reset()
    	episode_reward = 0
        
	
    	for t in range(5000):  # Adjust the time steps as needed
        	action = agent.get_action(state)
        	next_state, reward, done, _ = env.step(action)
        	agent.buffer.add(state, action, reward, next_state, done)

        	if len(agent.buffer.buffer) > agent.batch_size:
            		agent.train()

        	state = next_state
        	episode_reward += reward

        	if done:
           		break

    	print(f"Episode: {episode + 1}, Reward: {episode_reward}")
    env.close()


if __name__ == "__main__":
    
    

    env = gym.make('QuadrotorPlusHoverEnv-v0')
    # env = gym.make('QuadrotorXHoverEnv-v0')
    # env = gym.make('TiltrotorPlus8DofHoverEnv-v0')
    main(env)
