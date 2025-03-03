import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %ls *.csv

csv_file = "covariates_link_effect.csv"
data = pd.read_csv(csv_file, sep = "\t")
data = data.rename( columns = {"estimation":"accuracy"})

plt.figure(figsize=(10, 6))
ax = sns.boxplot(data = data,
            x = "p", y = "accuracy", hue = "method", showfliers = False)
sns.move_legend(ax, "upper left" , bbox_to_anchor=(1, 1))
plt.savefig("covariates_link_effect.png")

csv_file = "covariates_shift_assumption.csv"
data = pd.read_csv(csv_file, sep = "\t")
data = data.rename( columns = {"estimation":"accuracy"})

plt.figure(figsize=(10,6))
ax = sns.boxplot(data = data,
            x = "mB", y = "accuracy", hue = "method", showfliers = False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_xlabel("mB = 1:[0,0,0], 2:[1,0,0], 3:[1,1,0] , 4:[1,2,0]")
plt.savefig("covariates_shift_assumption.png")


