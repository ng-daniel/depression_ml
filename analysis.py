import pandas as pd
import random
import matplotlib.pyplot as plt
from data import preprocess_train_test_dataframes

data = pd.read_csv("data/processed_dataframes/data_resampled.csv", index_col=0)
data_control = data[data['label'] == 0].copy()
data_condition = data[data['label'] == 1].copy()

def get_time_index():
    time_index = [f'{12 + i//60:02d}' for i in range(0,720,120)]
    time_index += [f'{i//60:02d}' for i in range(0,720,120)]
    time_index += ['12']
    return time_index

def data_mean_med_std(data : pd.Series):
    d_mean = data.mean()
    d_median = data.median()
    d_std = data.std()
    return pd.Series([d_mean, d_median, d_std], ['mean', 'median', 'std'])

def data_mean_trend_plot(data_control:pd.DataFrame = None, data_condition:pd.DataFrame = None, stat:str = None, title=None):
    fig, ax = plt.subplots()

    time_index = get_time_index()    

    if data_control is not None:
        data_control = data_control.drop(labels=['label'], axis=1)
        data_control_trend = data_control.apply(data_mean_med_std, axis=0).transpose()
        markers, caps, bars = plt.errorbar(
            x=data_control_trend.index,
            y=data_control_trend[stat],
            yerr=data_control_trend['std'],
            label='Control', color='blue',
            errorevery=7, capsize=0
        )
        [bar.set_alpha(0.15) for bar in bars]
        [cap.set_alpha(0.15) for cap in caps]
    
    if data_condition is not None:
        data_condition = data_condition.drop(labels=['label'], axis=1)
        data_condition_trend = data_condition.apply(data_mean_med_std, axis=0).transpose()
        markers, caps, bars = plt.errorbar(
            x=data_condition_trend.index,
            y=data_condition_trend[stat],
            yerr=data_condition_trend['std'],
            label='Condition', color='red',
            errorevery=7, capsize=0
        )
        [bar.set_alpha(0.15) for bar in bars]
        [cap.set_alpha(0.15) for cap in caps]

    ax.set_xticks(ticks=range(0,1441,120), labels=time_index)
    plt.xlabel('Time(Hour of Day)')
    plt.ylabel('Number of Recorded Movements')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_random_samples(data_control:pd.DataFrame, data_condition:pd.DataFrame, n_random:int):    
    control_index = random.sample(range(len(data_control.index)), n_random)
    condition_index = random.sample(range(len(data_condition.index)), n_random)

    control_selected = data_control.iloc[control_index].copy()
    condition_selected = data_condition.iloc[condition_index].copy()

    fig, ax = plt.subplots()
    for i in range(n_random):
        plt.plot(control_selected.iloc[i], label='control', color='blue', alpha=0.5)
    for i in range(n_random):
        plt.plot(condition_selected.iloc[i], label='condition', color='red', alpha=0.5)
    
    time_index = get_time_index()
    ax.set_xticks(ticks=range(0,1441,120), labels=time_index)
    plt.xlabel('Time(Hour of Day)')
    plt.ylabel('Number of Recorded Movements')
    plt.title('I CANT TAKE IT ANYMORE!!!')
    plt.show()

# data_mean_trend_plot(data_control=data_control, 
#                      data_condition=data_condition, 
#                      stat='mean',
#                      title='Mean and Std. Dev Across All Raw Samples')

(processed_data, _) = preprocess_train_test_dataframes(
    X_train=data.drop(labels=['label'], axis=1),
    log_base=None,
    scale_range=None,
    use_gaussian=30,
    use_standard=False
)
processed_data['label'] = data['label']
data_control = processed_data[processed_data['label'] == 0].copy()
data_condition = processed_data[processed_data['label'] == 1].copy()
plot_random_samples(data_control.drop(labels=['label'], axis=1), data_condition.drop(labels=['label'], axis=1), 100)
data_mean_trend_plot(data_control=data_control, 
                     data_condition=data_condition, 
                     stat='mean',
                     title='Mean and Std. Dev Across All Preprocessed Samples')

# (processed_data, _) = preprocess_train_test_dataframes(
#     X_train=data.drop(labels=['label'], axis=1),
#     log_base=10,
#     use_gaussian=False,
#     use_standard=True
# )
# processed_data['label'] = data['label']
# data_control = processed_data[processed_data['label'] == 0].copy()
# data_condition = processed_data[processed_data['label'] == 1].copy()
# data_mean_trend_plot(data_control=data_control, 
#                      data_condition=data_condition, 
#                      stat='mean')