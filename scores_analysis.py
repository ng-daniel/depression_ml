import os
import pandas as pd
import random
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def plot_demographic_pyramid(scores, ax):
    ages = [f'{n}-{n+4}' for n in range(0,84,5)]
    ages_f = scores[scores['gender'] == 1]['age'].value_counts().sort_index()
    ages_m = scores[scores['gender'] == 2]['age'].value_counts().sort_index() / -1

    data = pd.DataFrame(index=ages)
    data['female'] = ages_f
    data['male'] = ages_m
    data.fillna(0)

    mbar = ax.barh(y=data.index, width=data['male'], label='male', color='#1f77b4')
    fbar = ax.barh(y=data.index, width=data['female'], label='female', color='orange')

    ax.bar_label(mbar, labels=[int(abs(val)) if val < 0 else 0 for val in data['male']])
    ax.bar_label(fbar)

    ax.set_ylabel('Age Ranges(Years)')
    ax.set_xlabel('# of Samples')
    ax.set_title('Population Pyramid of Depressed Subjects')
    ax.legend()

    return ax

def plot_education_marriage_work(scores, ax):
    education_ranges = scores[['edu', 'marriage', 'work']].copy()
    common_range = education_ranges['edu'].mode().item()
    regex='\d+-\d+'
    invalid_edu_mask = ~education_ranges['edu'].str.contains(regex, regex=True)
    education_ranges.loc[invalid_edu_mask, 'edu'] = common_range

    education_ranges['marriage'] = education_ranges['marriage'].map(lambda x : True if x == 1.0 else False)
    education_ranges['work'] = education_ranges['work'].map(lambda x : True if x == 1.0 else False)

    marriage_by_edu = education_ranges.groupby('edu')['marriage'].value_counts().unstack(level=-1)
    work_by_edu = education_ranges.groupby('edu')['work'].value_counts().unstack(level=-1)   

    data = pd.DataFrame(index=marriage_by_edu.index, columns=['no_marriage', 'marriage', 'no_work_study', 'work_study'])
    data['no_marriage'] = marriage_by_edu[False]
    data['marriage'] = marriage_by_edu[True]
    data['no_work_study'] = work_by_edu[False]
    data['work_study'] = work_by_edu[True]
    data.fillna(0.0, inplace=True)
    data = data.reindex(['6-10', '11-15', '16-20'])
    data.reset_index(inplace=True)

    x = np.arange(len(data['edu']))
    width = 0.8
    m0 = ax.bar(x,data['no_marriage'], width, label='no marriage', color='orange', alpha=0.3)
    m1 = ax.bar(x,data['marriage'], width, bottom=data['no_marriage'], label='marriage', color='orange')
    w0 = ax.bar(x,data['no_work_study'], width, bottom=data['no_marriage'] + data['marriage'], label='no work/school', color='#1f77b4', alpha=0.3)
    w1 = ax.bar(x,data['work_study'], width, bottom=data['no_marriage']+data['marriage']+data['no_work_study'], label='work/school', color='#1f77b4')

    ax.bar_label(m0, labels=[int(val) if val > 0 else ' ' for val in data['no_marriage']], label_type='center')
    ax.bar_label(m1, labels=[int(val) if val > 0 else ' ' for val in data['marriage']], label_type='center')
    ax.bar_label(w0, labels=[int(val) if val > 0 else ' ' for val in data['no_work_study']], label_type='center')
    ax.bar_label(w1, labels=[int(val) if val > 0 else ' ' for val in data['work_study']], label_type='center')

    ax.set_xlabel('Level of Education(Years)')
    ax.set_ylabel('# of Subjects')
    ax.set_title('Marital & Work/School Status by Education Level of Condition Class')
    ax.set_xticks(x)
    ax.set_xticklabels(data['edu'])
    ax.grid(which='both', alpha=0.4, color='lightgray')
    ax.minorticks_on()
    ax.legend()

scores = pd.read_csv("data/scores.csv")
scores = scores[scores['number'].str.contains('condition')]
print(scores)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
fig.tight_layout(pad=3)
plot_demographic_pyramid(scores, ax1)
plot_education_marriage_work(scores, ax2)
plt.show()
fig.savefig("figures/scores_demographics.png")

print("Done.")