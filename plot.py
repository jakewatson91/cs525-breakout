import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Plot:
    def __init__(self, y):
        
        self.df = pd.DataFrame({'Episode' : range(1, len(y) + 1),
                                'Metric' : y})
        
        self.df_filtered = self.df[self.df['Episode'] % 100 == 0]

    def plot(self, ylabel, title, figname, filename):
        plt.figure() # new figure
        sns.lineplot(data=self.df_filtered, x='Episode', y='Metric', linewidth=0.5)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(f"plots/{filename}_{figname}") if filename else plt.savefig(f"plots/{figname}")
        print(f"{figname} plot saved.")
        plt.show()