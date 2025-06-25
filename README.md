# Machine Learning for Detecting Depression from Motion Data

This repo is all the machine learning pipelines and code I wrote in Python for rapid experimentation of various machine learning models and preprocessing techniques. 
I learned so much through this project, and if you want the details and a couple figures, it's all here so go ham.

For the short version, this project is about relationship between physical movement patterns and depression, and trying to detect this with machine learning. The dataset, called "Depresjon", contains motion data from 55 subjects-- some depressed, some non-depressed(Enrique et al).

After confirming that there is at least *some* difference between the classes so I'm not completely wasting my time, I split the data using **5 fold** cross validation, **preprocessed** data for model input, **trained** various models on the data, and **evaluated** their respective performance. 

In the end, I was able to beat the "Depresjon" paper's baseline results, improving on both accuracy and f1score. It wasn't by much, but a win is a win :) and I've tinkered with this repo for way too long so I'm leaving it at that.

## You're Telling Me A Pipeline Preprocessed This Data?

The Depresjon Dataset consists of 55 subjects, with an average of 12-13 days of data each. The data is recorded in minute intervals. Here's a `.csv` snippet of the control_1 subject:

| timestamp | date | activity |
|-----------|------|----------|
| 2003-03-18 15:00:00 | 2003-03-18 | 60 |
| 2003-03-18 15:01:00 | 2003-03-18 | 0 |
| 2003-03-18 15:02:00 | 2003-03-18 | 264 |
| 2003-03-18 15:03:00 | 2003-03-18 | 662 |
| 2003-03-18 15:04:00 | 2003-03-18 | 293 |

The `timestamp` and `date` columns are self explanatory, but the `activity` column needs some more info. During data collection, subjects wore a motion traching watch that sampled at 32 hz for movements above 0.05 g, or ~0.5 m/s(Enrique et al). The `activity` column records the number of recorded movements for the duration of that minute.

Part of why I loved working on this dataset was that it's pretty challenging, for a few reasons:
1. **Small dataset**. With only ~700 labeled samples, the big-picture trends are harder to detect.
2. **Imbalanced Classes**. 32 control subjects v.s. only 23 condition subjects means models lean towards the majority class.

## References That I Definitely Didn't Make Up

1. Enrique Garcia-Ceja, Michael Riegler, Petter Jakobsen, Jim Tørresen, Tine Nordgreen, Ketil J. Oedegaard, and Ole Bernt Fasmer. 2018. Depresjon: a motor activity database of depression episodes in unipolar and bipolar patients. In Proceedings of the 9th ACM Multimedia Systems Conference (MMSys '18). Association for Computing Machinery, New York, NY, USA, 472–477. https://doi.org/10.1145/3204949.3208125

2. Ashford, J., Bird, J.J., Campelo, F., Faria, D.R. (2020). Classification of EEG Signals Based on Image Representation of Statistical Features. In: Ju, Z., Yang, L., Yang, C., Gegov, A., Zhou, D. (eds) Advances in Computational Intelligence Systems. UKCI 2019. Advances in Intelligent Systems and Computing, vol 1043. Springer, Cham. https://doi.org/10.1007/978-3-030-29933-0_37
