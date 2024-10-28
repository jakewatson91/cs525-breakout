try:
    import random
    print("Random module imported:", random.random())  # Generate a random float

    import numpy as np
    print("Numpy module imported:", np.array([1, 2, 3]))  # Create a numpy array

    from collections import deque
    d = deque([1, 2, 3])
    print("Collections deque imported:", d)  # Initialize a deque

    import os
    print("OS module imported:", os.getcwd())  # Get current working directory

    import sys
    print("Sys module imported:", sys.version)  # Print Python version

    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    print("PyTorch imported:", torch.__version__)  # Print PyTorch version

    from agent import Agent
    print("Agent imported successfully.")

    from dqn_model import DQN
    print("DQN model imported successfully.")

    from environment import Environment
    print("Environment imported successfully.")

    print("Done")

except ImportError as e:
    print("An import failed:", e)
