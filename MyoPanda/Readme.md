# DeepMyo

DeepMyo is a project of building deep learning classifier to identify different hand/wrist gestures.


The aim of the project is to leaverage Convoulutional Neural Network (CNN) to help improve the accuracy of classifying different hand/wrist gestures. 
We'll test the gesture classification accuracy both on patients / healthy participants. The scenerios of the classification will be further divided into:
1. Within session
2. Between sessions
3. Between users

The data preprocessing is referenced from study in 2019 "Deep Learning for Electromyographic Hand Gesture Signal Classification Using Transfer Learning" https://ieeexplore.ieee.org/document/8630679


## Side note
Currently, the data contains only the same subject, both in PretrainData and EvaluationData. More subjects will be added if possible and the PretrainData and EvaluationData will not contain the same subject

## Installation (Anaconda)

Create an environment with packages installation (Use requirements.txt)

After installing the Anaconda, open the Anaconda Prompt and enter the following commands:

conda create -name "env_name" python = 3.7 

conda activate "env_name"

conda install --file requirements.txt

## Usage

```python
# Read pickle dataframe, which contains EMG data of all subjects sorted by gestures. Each subjects have at least 2 sessions and 3 trials in one session. 
df = read_file()

# Preprocess data
X_train, y_train, _ = preprocess(df_subject_train, window_size = 52, nonoverlap_size = 5) = read_file(mode = 1 , window_size = 52, nonoverlap_size = 5)
X_val, y_val, _ = preprocess(df_subject_val, window_size = 52, nonoverlap_size = 5) = read_file(mode = 1 , window_size = 52, nonoverlap_size = 5)
X_test, y_test, _ = preprocess(df_subject_test, window_size = 52, nonoverlap_size = 5) = read_file(mode = 1 , window_size = 52, nonoverlap_size = 5)


```
