import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

from core.eval import create_metrics_table, metric_class_averages

RESULTS_DIR = 'results'
FIG_DIR = "figures"
FILENAME = "final_results.csv"

def plot_class_comparisons(results:pd.DataFrame, class_val, ax, title, cbarlabel, colormap):
    
    results = results[(results['note']==class_val) & (results['model_name']!='ZeroR')]
    classes = list(results['model_name'])
    metrics = [col for col in results.columns if (col != 'model_name' and col != 'note')]
    values = np.array(results.loc[:,metrics])
    
    im = ax.imshow(values, cmap=colormap)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.07, pad=0.05)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(range(len(metrics)), labels=metrics, rotation=0, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(classes)), labels=classes, rotation=60, ha="right", rotation_mode="anchor")

    threshold = im.norm(values.max())/1.6
    textcolors = ("black", "white")
    for i in range(len(classes)):
        for j in range(len(metrics)):
            color=textcolors[int(im.norm(values[i, j]) > threshold)]
            fontweight = 'bold' if values[i,j] == values[:,j].max() else None
            text = ax.text(j, i, values[i, j], ha="center", va="center", color=color, fontweight=fontweight)

    ax.set_title(title)

# aggregate all final results into a single dataframe
results = []

filenames = os.listdir(RESULTS_DIR)
for filename in filenames:
    if filename == FILENAME:
        continue
    model_results = pd.read_csv(os.path.join(RESULTS_DIR, filename), index_col=0)
    model_results = metric_class_averages(model_results[model_results['note']=='wt_avg'], weight_support=False)
    results.append(model_results)

final_results = pd.concat(results).reset_index(drop=True)
final_results = final_results.drop(labels=['sup', 'loss'], axis=1)
final_results = final_results.map(lambda x : round(x,3) if isinstance(x,float) else x)
final_results.to_csv(os.path.join(RESULTS_DIR, FILENAME))
print(final_results)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,6.5))
fig.tight_layout(pad=4)
plot_class_comparisons(final_results, class_val=0, ax=ax1, title="Non-Depressed", cbarlabel="Metric Value", colormap='PuBu')
plot_class_comparisons(final_results, class_val=1, ax=ax2, title="Depressed", cbarlabel="Metric Value", colormap='YlOrRd')
plot_class_comparisons(final_results, class_val='wt_avg', ax=ax3, title="Macro Average.", cbarlabel="Metric Value", colormap='RdPu')
fig.savefig(os.path.join(FIG_DIR, "heatmaps_macro.png"))
plt.show()