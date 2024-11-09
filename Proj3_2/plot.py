import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Plot:
    def __init__(self, y):
        
        self.df = pd.DataFrame({'Episode' : range(1, len(y) + 1),
                                'Metric' : y})

    def plot(self, ylabel, title, figname, filename):
        plt.figure() # new figure
        sns.lineplot(data=self.df, x='Episode', y='Metric')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(f"plots/{filename}_{figname}") if filename else plt.savefig(f"plots/{figname}")
        print(f"{figname} plot saved.")
        plt.show()






