"""Script to verify uniformity of p-value distribution in LANL dataset"""
# %% Import
import pandas as pd
from scipy.stats import probplot
import matplotlib.pyplot as plt

# %% Load p-values
py = pd.read_table(r"C:\Users\user\Desktop\phase_4_source_score_py.txt", names=("source", "score", "timewhen"))
dp = pd.read_table(r"C:\Users\user\Desktop\phase_4_source_score_dp.txt", names=("source", "score", "timewhen"))
ddcrp = pd.read_table(r"C:\Users\user\Desktop\phase_4_source_score_ddcrp.txt", names=("source", "score", "timewhen"))
stream_py = pd.read_table(r"C:\Users\user\Desktop\phase_4_source_score_streampy.txt", names=("source", "score", "timewhen"))

# %% Plot
fig, ax = plt.subplots(1)
probplot(py.score, dist="uniform", fit=False, plot=ax)
probplot(dp.score, dist="uniform", fit=False, plot=ax)
probplot(ddcrp.score, dist="uniform", fit=False, plot=ax)
probplot(stream_py.score, dist="uniform", fit=False, plot=ax)
ax.plot([0, 1], [0, 1], "--", color="red")
ax.get_lines()[0].set_color('blue')
ax.get_lines()[0].set_markersize(1)
ax.get_lines()[1].set_color('cyan')
ax.get_lines()[1].set_markersize(1)
ax.get_lines()[2].set_color('green')
ax.get_lines()[2].set_markersize(1)
ax.get_lines()[3].set_color('magenta')
ax.get_lines()[3].set_markersize(1)
ax.get_lines()[3].set_color('red')
ax.grid()
ax.legend(("PY", "DP", "DDCRP", "STREAM PY (36h)"))
fig.savefig("./plots/qqplot.pdf")

# %%
