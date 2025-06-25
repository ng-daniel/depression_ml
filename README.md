# Machine Learning for Detecting Depression from Motion Data

This repo is all the machine learning pipelines and code I wrote in Python for rapid experimentation of various machine learning models and preprocessing techniques. 
I learned so much through this project, and if you want the details and a couple figures, it's all here (I promise to keep it brief!).

For the short version, this project is about relationship between physical movement patterns and depression, and trying to detect this with machine learning. The dataset, called "Depresjon", contains motion data from 55 subjects-- some depressed, some non-depressed(Enrique). After confirming that there is at least *some* difference between the classes so I'm not completely wasting my time, I split the data using 5 fold cross validation, preprocessed data for model input, trained various models on the data, and evaluated their respective performance. In the end, I was able to improve on the 

which led me to the actual paper that published the data(Enrique),  

learning. Additionally, this was an opportunity for me to refine my skills in machine learning.


I found this dataset on Kaggle, and it attracted me because the 

## References That I Definitely Didn't Make Up

1. Enrique Garcia-Ceja, Michael Riegler, Petter Jakobsen, Jim Tørresen, Tine Nordgreen, Ketil J. Oedegaard, and Ole Bernt Fasmer. 2018. Depresjon: a motor activity database of depression episodes in unipolar and bipolar patients. In Proceedings of the 9th ACM Multimedia Systems Conference (MMSys '18). Association for Computing Machinery, New York, NY, USA, 472–477. https://doi.org/10.1145/3204949.3208125

2. Ashford, J., Bird, J.J., Campelo, F., Faria, D.R. (2020). Classification of EEG Signals Based on Image Representation of Statistical Features. In: Ju, Z., Yang, L., Yang, C., Gegov, A., Zhou, D. (eds) Advances in Computational Intelligence Systems. UKCI 2019. Advances in Intelligent Systems and Computing, vol 1043. Springer, Cham. https://doi.org/10.1007/978-3-030-29933-0_37
