# DeepMyo

DeepMyo is a project of building deep learning classifier to identify different hand/wrist gestures.

Dataset: The dataset is separated in two subdatasets (pre-training and evaluation dataset). 
The folder training0 (for the pre-training dataset) and the folders retraining and test0 (for the evaluation dataset) contain the raw myo armband signal in files named subject_num_WG_Trail_i where WG means with gravity and i goes from 0 to 7. 
Each file contain a the sEMG signal for a specific gestures. In order: 0 = Fist, 1 = Fist with wrist ext., 2 = Open palm, 3 = Open palm with wrist ext., 4 = Opposition, 5 = Lateral pinch, 6 = Cylinder grip, 7 = Relax. 
Eight gestures forming a cycle, and 4 cycles forming a round (PretrainData has 1 round and Evaluation has 2 rounds, where the first round is used for retraining/calibration). 
Note that PretrainingData and EvaluationData were collected in different sessions (Myo armband has been removed in between, whereas two rounds in EvaluationData were in the same session).

## Installation

Create an environment with packages installation (Use requirements.txt)

Anaconda:
conda create -name "env_name" python = 3.7 

conda activate "env_name"

conda install --file requirements.txt

## Usage

```python
# Read pretrain data
X_pretrain, y_pretrain = read_file(mode = 0 , window_size = 52, nonoverlap_size = 5)

# Read retrain data
X_retrain, y_retrain = read_file(mode = 1 , window_size = 52, nonoverlap_size = 5)

# Read pretrain data
X_test, y_test = read_file(mode = 2 , window_size = 52, nonoverlap_size = 5)

# Calculate the accuracies across different gestures
acc_dict = class_accuracy(y_pred, y_true)

# Plot the barplot of acc_dict
gesture_list = ['Fist', 'Fist with wrist ext.', 'Open palm', 'Open palm with wrist ext.', 'Opposition', 'Lateral Pinch', 'Cylinder Grip', 'Relax']
class_accuracy_plot(acc_dict, gesture_list)
```
