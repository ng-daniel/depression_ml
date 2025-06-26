import os
import pandas as pd
import random
import matplotlib.pyplot as plt

from core.eval import create_metrics_table, metric_class_averages

def plot_metric(results:pd.DataFrame, metric:str, ax):
    metric_results = results[['name', metric]]



RESULTS_DIR = 'results'
FILENAME = "final_results.csv"

# aggregate all final results into a single dataframe
results = []

filenames = os.listdir(RESULTS_DIR)
for filename in filenames:
    if filename == FILENAME:
        continue
    model_results = pd.read_csv(os.path.join(RESULTS_DIR, filename), index_col=0)
    model_results = metric_class_averages(model_results[model_results['note']=='wt_avg'])
    print(model_results)
    results.append(model_results)

# final_results = create_metrics_table(results)
# final_results.to_csv(os.path.join(RESULTS_DIR, FILENAME))
# print(final_results)
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(8,8))
