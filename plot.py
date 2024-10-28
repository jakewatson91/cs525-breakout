import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Plot:
    def __init__(self, avg_rewards):

        self.avg_rewards = avg_rewards
        self.df = pd.DataFrame({'Episode' : range(1, len(avg_rewards) + 1),
                                'Avg_Rewards' : avg_rewards})

    def plot(self):
        sns.lineplot(data=self.df, x='Episode', y='Avg_Rewards')
        plt.show()
        plt.savefig("Training.png")
        print("plot saved")





