# Machine Learning for Detecting Depression from Motion Data

This repo contains all the machine learning pipelines and code I wrote in Python for rapid experimentation of various machine learning models and preprocessing techniques. This README details my process for this project with explanations for certain techniques and figures of results and other interesting findings.

### The Short Version

This project is about relationship between physical movement patterns and depression, and trying to detect this with machine learning. The dataset, called Depresjon, contains motion data from 55 subjects-- some depressed, some non-depressed(Enrique et al). This time-series dataset was challenging to work with because it is small, imbalanced, and full of noise.

After confirming that there is at least _some_ difference between the classes in the data so I'm not completely wasting my time, I...

1. Split the data
2. Preprocessed data using various techniques for model input
3. Trained several models on the data
4. Evaluated their respective performances

In the end, I was able to beat the Depresjon paper's results, improving on multiple key metrics. Though there may still be room for improvement, I've tinkered with this project for way too long so I'm leaving it at that :).

Future work could involve exploring addiional features for the data or to develop a lightweight application where users can upload their own data and figure out their likelihood of having depression based on the best performing model.

## Repo Guide

```
depression
├───core                    # core functions for preprocessing,
│                             training, evaluation, etc.
│
├───data                    # .csv data files from Depresjon
│   ├───condition
│   ├───control
│   └───processed_dataframes    # reformatted samples
│       └───kfolds                  # .txt files with sample names
│                                     for cross val folds
│
├───figures                 # figures, graphs, charts and stuff
│───results                 # .csv files of experiment results
│
│─data_analysis.py      # visualizing samples and preprocessing
│─load_data.py          # aggregating samples and creating folds
│─results.py            # formatting and visualizing results
│─scores_analysis.py    # visualizing subject demographics
│
│─train_A.py            # training modules for respective models
...
└─train_Z.py
```

### Dependencies

- Numpy
- Pandas
- Pytorch
- Sci-kit Learn
- Scipy
- Xgboost
- imbalanced-learn

### How to Run

1. Fork and clone repo
2. Install dependencies
3. Run a training module or the other programs (`analysis`, `load_data`, `results`)
   - Adjust the preprocessing settings and model hyperparameters if you like.
4. If you really want to test a new, different model:
   1. copy the format of an existing training module, except with your Pytorch/Sci-kit Learn model
   2. create a new training loop in the `training_loops.py`

## Data Details

The Depresjon Dataset consists of 55 subjects, with an average of 12-13 days of data each. The data is recorded in minute intervals. Here's a `.csv` snippet of subject `control_1`:

| timestamp           | date       | activity |
| ------------------- | ---------- | -------- |
| 2003-03-18 15:00:00 | 2003-03-18 | 60       |
| 2003-03-18 15:01:00 | 2003-03-18 | 0        |
| 2003-03-18 15:02:00 | 2003-03-18 | 264      |
| 2003-03-18 15:03:00 | 2003-03-18 | 662      |
| 2003-03-18 15:04:00 | 2003-03-18 | 293      |

During data collection, subjects wore a motion traching watch that sampled at 32 hz for movements above 0.05 g, or ~0.5 m/s(Enrique et al). The `activity` column records the number of recorded movements for the duration of that minute.

The dataset was nice enough to provide some demographic information, but only for the condition/depressed class. Regardless, I've whipped up some quick visuals for them, as they can still tell us about the population we're working with.

![2 figures depicting condition subject demographics](./figures/scores_demographics.png)

It seems like the age range is pretty wide, with most subjects in their 30s and 40s. There's a healthy mix of married and unmarried individuals, but most subjects are not currently employed or in school.

## You're Telling Me A Pipeline Preprocessed This Data?

### Challenges with the dataset:

1. **Small Dataset**. With only ~700 labeled samples and 1440 datapoints each, the big-picture trends are harder to detect.
2. **Imbalanced Classes**. 32 control subjects v.s. only 23 condition subjects means models lean towards the majority class.
3. **Noisy Data**. On a minute level, movement patterns throughout the day are very erratic, which hinder model predictions.

### Preprocessing Pipeline

1. **SMOTE Resampling**. To alleviate class imbalance, Synthetic Minority Oversampling Technique is used to even out the class distributions in the training set by reducing the majority class and increasing the minority class. According to the original paper, SMOTE generates samples by "drawing lines" between k-nearest-neighbors in feature space, selecting new samples along these lines(Chawla et al).

2. **Scaling**. Models, especially neural nets, dislike big numbers, and our data is full of big numbers. Several scaling techniques are supported, specifically log scaling, min-max scaling, and standard scaling. Scaling operations are first fitted to the training data, then applied to training and evaluation data.

3. **De-Noising**. The data is very erratic due to it's relatively high sample rate. Since I'm not doing any foreacasting and care more about overall movement trends rather than the minute details, it proved beneficial to introduce some denoising to smoothen out the time series, eliminating sudden peaks and outliers. Gaussian de-noising was specifically selected for its simplicity and effectiveness on random noise.

4. **Seasonal Adjustments**. Looking at the data reveals a clear circadian cycle, as one would expect. Removing this large pattern might reveals subtle differences in the classes that otherwise would have gone unnoticed, improving model performance. I achieve this by fitting a polynomial to the training data, then subtracting it from all samples.

<sup>\*Not all preprocessing techniques were used for each model, as the optimal processing setup is different for each one.</sup>

Below are some figures visualizing a possible preprocessing result. On the left, the solid and transparent colors represent the mean and standard deviation, respectively, of all samples by minute. The right shows individual data points from 10 samples, 5 per class.

![several red and blue time-series figures depicting the preprocessing pipeline](./figures/preprocessing.png)

From these figures, the day/night pattern is pretty evident. You may also notice that all graphs start at 12PM. That's because the samples needed to follow the same circadian rhythym, and 12 seemed like a reasonable hour of day. Applying scaling and de-noising additionally reveals some divots in the data at certain hours of the day. They could represent common meal times in Norway, where this dataset was collected.

### Feature Extraction

Since I was working on a similar project at the time, I got some ideas for potential features from a paper about extracting features from EEG data, including the sliding window technique(Ashford et al).

<sup>\*One idea involved features from a fourier-transform of the data, which makes sense for EEG data. It made predictions worse for my data, but at least I confirmed that it wasn't useful.</sup>

1. **Simple Statistical Features**. Standard statistics like means, medians, maximums and minimums to get a big picture view of the data.

2. **Sliding Window Approach**. Rather than just calculate a few features for the whole sample, by sliding that window across the data and taking snapshots along the way, we can extract way more features while also taking into account the flow of time. and it works pretty well for some models.

<sup>\*Similar to the preprocessing techniques, different models had different extracted features, ie. different window sizes or types of features used.</sup>

## Model Selection, Training, and Evaluation

Brief show and tell synopsis of the models I selected and why. Go to each model's respective training module if you want to see the specific preprocessing settings I found to work best. Visit `core/model.py` to see the neural network architecture.

### Machine Learning Models

| name                          | python class name      | description                                     |
| ----------------------------- | ---------------------- | ----------------------------------------------- |
| Zero Rule Baseline            | ZeroR                  | Only predicts the majority class.               |
| Random Forest                 | RandomForestClassifier | Solid ensemble model.                           |
| XGBoost                       | XGBClassifier          | Random Forest's gradient boosted little cousin. |
| Linear Support Vector Machine | SVC                    | Best performing Depresjon baseline.             |

### Neural Network Architecture Models

| name                        | python class name | description                                                          |
| --------------------------- | ----------------- | -------------------------------------------------------------------- |
| Multilayer Perceptron (MLP) | FeatureMLP        | Basic neural network architecture.                                   |
| 1D CNN                      | ConvNN            | Lightweight feature extraction.                                      |
| LSTM                        | LSTM              | Type of RNN, great for time series data.                             |
| CNN LSTM Hybrid             | ConvLSTM          | 1d CNN with more serious temporal shenanigans.                       |
| Feature LSTM                | LSTM_Feature      | LSTM model with sliding window features instead of pure motion data. |

### Evaluation Metrics

Though I used SMOTE earlier to address the class imbalance in the training set, the evaluation dataset is still imbalanced. Therefore, it is still important that we select metrics that are resistant to imbalanced classes(thanks Depresjon for the metric suggestions).

| name      | abreviaton | range  | description                                                                           |
| --------- | ---------- | ------ | ------------------------------------------------------------------------------------- |
| Accuracy  | acc        | [0,1]  | % of correct predictions.                                                             |
| Precision | prec       | [0,1]  | % of correct positive predictions out of all positive predictions.                    |
| Recall    | rec        | [0,1]  | % of actual positives identified.                                                     |
| F1-Score  | f1sc       | [0,1]  | Harmonic mean of precision and recall.                                                |
| MCC       | mcc        | [-1,1] | Matthews Correlation Coefficient, takes into account T/F positives and T/F negatives. |

With the main goal being identifying depressed individuals from motion data, the ideal metrics would **maximize recall on the depressed class without over-reducing other metrics** like F1-Score or the MCC.

<sup>\*Even with 5 fold cross validation, all neural networks saw significant instability in evaluation metrics between identical experiments due to their nondeterministic nature. To fix this, I ran an additional 29 trials for each neural network and averaged the results to better represent their true performance.</sup>

## Results

Final metrics of 5-Fold cross validation. The models in the tables and graphs are not ordered in any particular manner. **Best metrics are in bold.**

---

### _Non-Depressed Class_

| model_name             | acc       | prec      | rec       | f1sc      | mcc       |
| ---------------------- | --------- | --------- | --------- | --------- | --------- |
| ConvNN                 | **0.733** | **0.795** | 0.713     | 0.747     | **0.463** |
| ConvLSTM               | 0.719     | 0.786     | 0.721     | 0.737     | 0.433     |
| SVC                    | 0.73      | 0.78      | 0.754     | **0.764** | 0.45      |
| LSTM                   | 0.71      | 0.788     | 0.676     | 0.722     | 0.419     |
| LSTM_Feature           | 0.723     | 0.766     | 0.755     | 0.751     | 0.435     |
| FeatureMLP             | 0.718     | 0.767     | 0.749     | 0.747     | 0.421     |
| RandomForestClassifier | 0.705     | 0.734     | **0.776** | 0.752     | 0.388     |
| XGBClassifier          | 0.708     | 0.743     | 0.761     | 0.749     | 0.402     |
| ZeroR                  | 0.562     | 0.567     | 1.0       | 0.715     | 0.0       |

---

### _Depressed Class_

| model_name             | acc       | prec      | rec       | f1sc      | mcc       |
| ---------------------- | --------- | --------- | --------- | --------- | --------- |
| ConvNN                 | **0.733** | 0.669     | **0.751** | **0.699** | **0.463** |
| ConvLSTM               | 0.719     | 0.655     | 0.711     | 0.671     | 0.433     |
| SVC                    | 0.73      | **0.695** | 0.708     | 0.698     | 0.45      |
| LSTM                   | 0.71      | 0.641     | 0.746     | 0.682     | 0.419     |
| LSTM_Feature           | 0.723     | 0.654     | 0.674     | 0.66      | 0.435     |
| FeatureMLP             | 0.718     | 0.661     | 0.662     | 0.656     | 0.421     |
| RandomForestClassifier | 0.705     | 0.677     | 0.608     | 0.637     | 0.388     |
| XGBClassifier          | 0.708     | 0.683     | 0.636     | 0.654     | 0.402     |
| ZeroR                  | 0.562     | 0.0       | 0.0       | 0.0       | 0.0       |

---

### _Macro(Unweighted) Average_

| model_name             | acc       | prec      | rec       | f1sc      | mcc       |
| ---------------------- | --------- | --------- | --------- | --------- | --------- |
| ConvNN                 | **0.733** | 0.732     | **0.732** | 0.723     | **0.463** |
| ConvLSTM               | 0.719     | 0.72      | 0.716     | 0.704     | 0.433     |
| SVC                    | 0.73      | **0.738** | 0.731     | **0.731** | 0.45      |
| LSTM                   | 0.71      | 0.714     | 0.711     | 0.702     | 0.419     |
| LSTM_Feature           | 0.723     | 0.71      | 0.715     | 0.706     | 0.435     |
| FeatureMLP             | 0.718     | 0.714     | 0.706     | 0.701     | 0.421     |
| RandomForestClassifier | 0.705     | 0.706     | 0.692     | 0.695     | 0.388     |
| XGBClassifier          | 0.708     | 0.713     | 0.699     | 0.701     | 0.402     |
| ZeroR                  | 0.562     | 0.284     | 0.5       | 0.358     | 0.0       |

---

### _Weighted Average_

| model_name             | acc       | prec      | rec       | f1sc      | mcc       |
| ---------------------- | --------- | --------- | --------- | --------- | --------- |
| ConvNN                 | **0.733** | 0.741     | 0.73      | 0.727     | **0.463** |
| ConvLSTM               | 0.719     | 0.729     | 0.717     | 0.709     | 0.433     |
| SVC                    | 0.73      | **0.744** | **0.734** | **0.735** | 0.45      |
| LSTM                   | 0.71      | 0.724     | 0.706     | 0.705     | 0.419     |
| LSTM_Feature           | 0.723     | 0.718     | 0.721     | 0.712     | 0.435     |
| FeatureMLP             | 0.718     | 0.721     | 0.712     | 0.708     | 0.421     |
| RandomForestClassifier | 0.705     | 0.71      | 0.704     | 0.703     | 0.388     |
| XGBClassifier          | 0.708     | 0.718     | 0.707     | 0.708     | 0.402     |
| ZeroR                  | 0.562     | 0.324     | 0.571     | 0.408     | 0.0       |

---

These metrics are better visualized with heatmaps. **Best metrics are in bold.**

![4 labeled heatmaps depicting model results](./figures/heatmap_results.png)

<sup>\*The results discussion will reference the original Depresjon paper results quite a bit, so use the links below to see the original paper and their results.</sup>

Overall, the best performing classifier with unweighted macro averages is the 1d CNN, with a MCC of **0.463** and an accuracy of **73.3%**. When weighting the averages by the number of samples in each class, the linear SVM ends up on top instead, improving in all metrics compared to the previous with **0.744** precision and **0.734** recall.

### Metrics Analysis

Models generally struggled predicting the depressed class more than the non-depressed class, which the Depresjon paper mentioned regarding their baseline experiments. The SMOTE technique proved to be successful in addressing this issue. Though SMOTE caused a slight drop in non-depressed precision, there were larger improvements in depressed precision, sometimes by over 10%. As a result, compared to the Depresjon paper's results, models using SMOTE achieved a much more balanced result across both classes.

An interesting phenomenon that I noticed between the two classes is the dichotomy between precision and recall. Models with low recall typically had high precision, and vice versa. Additionally, models with higher precision for the control class had higher recall in the condition class. This makes sense, as precision and recall are naturally against eachother. All else being equal, the more you focus on capturing an entire population(recall), the less likely you are to have every prediction be correct(precision). Furthermore, capturing the majority of one class may lead to capturing less of the other.

### Challenges

These metrics indicate that my models and data preprocessing techniques are an improvement over the original results based on both overall performance and depressed class performance. However, it is also clear that these results aren't perfect. Due to the small dataset size, neural network based models like LSTMs weren't able to perform at their highest capacity, since they are known to scale significantly with data availability. Additionally, as visualized in the preprocessing section, the difference between the two classes - though evident - is not extreme. Perhaps there just isn't enough of a difference between the two classes to achieve better results.

This makes sense, since there are many other factors that contribute to movement patterns than just depression. From the visualizations in the demographics section, we can see that the subject pool contains a diverse group of individuals, which all contribute to differences in daily movement patterns(for example, students may tend to stay up late more compared to the elderly). For future research, it may be worthwhile to collect data from a single demographic such as college students, trading the population scope for better predictions.

In conclusion, despite the challenges presented by this dataset, by using various preprocessing techniques and model architectures, I was able to improve on the paper's baseline results in classifying depressive states from motion data, especially performance on the depressed class.

Okay that's it byee!

## References That I Definitely Didn't Make Up

1. Enrique Garcia-Ceja, Michael Riegler, Petter Jakobsen, Jim Tørresen, Tine Nordgreen, Ketil J. Oedegaard, and Ole Bernt Fasmer. 2018. Depresjon: a motor activity database of depression episodes in unipolar and bipolar patients. In Proceedings of the 9th ACM Multimedia Systems Conference (MMSys '18). Association for Computing Machinery, New York, NY, USA, 472–477. https://doi.org/10.1145/3204949.3208125

2. Chawla, Nitesh V., Kevin W. Bowyer, Lawrence O. Hall, and W. Philip Kegelmeyer. "SMOTE: synthetic minority over-sampling technique." Journal of artificial intelligence research 16 (2002): 321-357. https://doi.org/10.1613/jair.953

3. Ashford, J., Bird, J.J., Campelo, F., Faria, D.R. (2020). Classification of EEG Signals Based on Image Representation of Statistical Features. In: Ju, Z., Yang, L., Yang, C., Gegov, A., Zhou, D. (eds) Advances in Computational Intelligence Systems. UKCI 2019. Advances in Intelligent Systems and Computing, vol 1043. Springer, Cham. https://doi.org/10.1007/978-3-030-29933-0_37
