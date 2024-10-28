# import torch
# import numpy as np
# from agent_dqn import Agent_DQN
from environment import Environment
# from argument import Args  # Assuming arguments are defined in a separate file

# Mock argument class if not available
# args = Args()
# args.lr = 0.001  # Learning rate
# args.test_dqn = False  # Set to True to load a pre-trained model if needed

# Initialize environment (assuming it's an Atari-like environment)
env = Environment(env_name="BreakoutNoFrameskip-v4", 
                  args=None, 
                  atari_wrapper=True)

# Initialize the agent
# agent = Agent_DQN(env, args)
state = env.reset()    

print("State shape before DQN:", state.shape)  # Should be [batch_size, channels, height, width]


# Test Replay Buffer - Add a dummy experience to the buffer
# state = np.zeros((84, 84, 4))  # Mock state (4 stacked frames)
# action = 1  # Example action
# reward = 1  # Example reward
# next_state = np.zeros((84, 84, 4))  # Mock next state
# done = False  # Example done flag

# # Push experience to the buffer
# print("Testing push()...")
# agent.push(state, action, reward, next_state, done)
# print("Replay Buffer:", len(agent.D), "experiences stored.")

# # Test Action Selection - Generate an action using the epsilon-greedy policy
# print("Testing make_action()...")
# selected_action = agent.make_action(state)
# print("Action selected:", selected_action)

# # Optional: Test Train Method (Add print statements inside train method for debugging)
# # print("Testing train()...")
# # agent.train()  # Uncomment once ready for training
