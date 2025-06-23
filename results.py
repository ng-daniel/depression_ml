import os
import pandas as pd
import random
import matplotlib.pyplot as plt

from core.eval import create_metrics_table

RESULTS_DIR = '../results'

# aggregate all final results into a single dataframe
results = []

filenames = os.listdir(RESULTS_DIR)
for filename in filenames:
    results.append(pd.read_csv(os.path.join(RESULTS_DIR, filename)))

final_results = create_metrics_table(results)
final_results.to_csv(os.path.join(RESULTS_DIR, "final_results.csv"))
print(final_results)

plt.bar