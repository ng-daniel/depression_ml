import os
import pandas as pd
import random
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from core.data import preprocess_train_test_dataframes
from core.util import data_mean_med_std

SEED = random.randint(0,32768)
DIR = "figures"

def get_time_index():
    '''
    Creates an index of times for every minute starting from 12:00 PM to 11:59 AM, then 12:00 PM again.
    Used as an X axis for graphs. 
    '''
    time_index = [f'{12 + i//60:02d}' for i in range(0,720,120)]
    time_index += [f'{i//60:02d}' for i in range(0,720,120)]
    time_index += ['12']
    return time_index

def data_mean_trend_plot(data_control:pd.DataFrame = None, data_condition:pd.DataFrame = None, stat:str = None, title=None, ax=None, ylabel:str=None):
    
    fig=None
    if ax is None:
        fig, ax = plt.subplots()

    time_index = get_time_index()    
    if data_control is not None:
        data_control_trend = data_control.apply(data_mean_med_std, axis=0).transpose()
        _, caps, bars = ax.errorbar(
            x=data_control_trend.index,
            y=data_control_trend[stat],
            yerr=data_control_trend['std'],
            label='non-depressed', color='blue',
            errorevery=7, capsize=0
        )
        [bar.set_alpha(0.15) for bar in bars]
        [cap.set_alpha(0.15) for cap in caps]
    
    if data_condition is not None:
        data_condition_trend = data_condition.apply(data_mean_med_std, axis=0).transpose()
        _, caps, bars = ax.errorbar(
            x=data_condition_trend.index,
            y=data_condition_trend[stat],
            yerr=data_condition_trend['std'],
            label='depressed', color='red',
            errorevery=7, capsize=0
        )
        [bar.set_alpha(0.15) for bar in bars]
        [cap.set_alpha(0.15) for cap in caps]

    ax.set_xticks(ticks=range(0,1441,120), labels=time_index)
    ax.set_xlabel('Time(Hour of Day)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    return (fig, ax)

def plot_random_samples(data_control:pd.DataFrame, data_condition:pd.DataFrame, n_random:int, ax=None, ylabel:str=None):    
    random.seed(SEED)
    control_index = random.sample(range(len(data_control.index)), n_random)
    condition_index = random.sample(range(len(data_condition.index)), n_random)

    control_selected = data_control.iloc[control_index].copy()
    condition_selected = data_condition.iloc[condition_index].copy()

    fig=None
    if ax is None:
        fig, ax = plt.subplots()

    for i in range(n_random):
        ax.scatter(range(0,1440), control_selected.iloc[i], label='non-depressed', color='blue', alpha=0.2, s=1.5)
    for i in range(n_random):
        ax.scatter(range(0,1440), condition_selected.iloc[i], label='depressed', color='red', alpha=0.2, s=1.5)
    
    time_index = get_time_index()    
    ax.set_xticks(ticks=range(0,1441,120), labels=time_index)
    ax.set_xlabel('Time(Hour of Day)')
    ax.set_ylabel(ylabel)
    ax.set_title(f' Activity Plot of {n_random*2} Samples')

    red_patch = mpatches.Patch(color='red', label='non-depressed')
    blue_patch = mpatches.Patch(color='blue', label='depressed')
    ax.legend(handles=[red_patch, blue_patch])

    return (fig, ax)

data = pd.read_csv("data/processed_dataframes/data_raw.csv", index_col=0)
data_control = data[data['label'] == 0].copy().drop(labels=['label'], axis=1)
data_condition = data[data['label'] == 1].copy().drop(labels=['label'], axis=1)

NUM_RANDOM = 5
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12,12))
fig.tight_layout(pad=3)
data_mean_trend_plot(data_control=data_control, 
                     data_condition=data_condition, 
                     stat='mean',
                     title='No Preprocessing',
                     ax=ax1,
                     ylabel='# of Recorded Movements')
plot_random_samples(data_control, 
                    data_condition, 
                    n_random=NUM_RANDOM, 
                    ax=ax2,
                    ylabel='# of Recorded Movements')

preprocessing_settings = {
    'resample' : False,
    'log_base' : None,
    'scale_range' : (0,1),
    'use_standard' : None,
}
(processed_data, _) = preprocess_train_test_dataframes(
    X_train=data.drop(labels=['label'], axis=1),
    settings=preprocessing_settings
)
processed_data['label'] = data['label']
data_control = processed_data[processed_data['label'] == 0].copy().drop(labels=['label'], axis=1)
data_condition = processed_data[processed_data['label'] == 1].copy().drop(labels=['label'], axis=1)
data_mean_trend_plot(data_control=data_control, 
                     data_condition=data_condition, 
                     stat='mean',
                     title='Scaled Data',
                     ax=ax3,
                     ylabel='Value')
plot_random_samples(data_control, 
                    data_condition, 
                    n_random=NUM_RANDOM, 
                    ax=ax4,
                    ylabel='Value')

preprocessing_settings = {
    'resample' : True,
    'log_base' : None,
    'scale_range' : (0,1),
    'use_standard' : True,
    'use_gaussian' : 30,
    'adjust_seasonality' : True,
}
(processed_data, _) = preprocess_train_test_dataframes(
    X_train=data.drop(labels=['label'], axis=1),
    settings=preprocessing_settings
)
processed_data['label'] = data['label']
data_control = processed_data[processed_data['label'] == 0].copy().drop(labels=['label'], axis=1)
data_condition = processed_data[processed_data['label'] == 1].copy().drop(labels=['label'], axis=1)
data_mean_trend_plot(data_control=data_control, 
                     data_condition=data_condition, 
                     stat='mean',
                     title='De-Noised Adjusted Data',
                     ax=ax5,
                     ylabel='Value')
plot_random_samples(data_control, 
                    data_condition, 
                    n_random=NUM_RANDOM, 
                    ax=ax6,
                    ylabel='Value')

fig.savefig(os.path.join(DIR, "preprocessing.png"))
plt.show()