import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def class_accuracy(y_pred, y_true):
    keys = [str(x) for x in list(np.arange(0, len(np.unique(y_true))))]
    acc_dict = {key: [] for key in keys}
    for class_label in np.unique(y_true):
        index = np.where((y_true == class_label))[0]
        y_pred_class, y_true_class = y_pred[index], y_true[index]
        correct_ind = np.where((y_pred_class == y_true_class))[0]
        acc = len(correct_ind)/(len(y_true_class))
        acc_dict[str(class_label)].append(np.round(acc, 2))
    return acc_dict

def class_accuracy_plot(acc_dict, gesture_list):
    data = pd.DataFrame(None, columns = ['Gesture', 'Accuracy'])
    data['Gesture'] = gesture_list
    data['Accuracy'] = [x[0] for x in list(acc_dict.values())]
    sns.set(rc={'figure.figsize':(12, 7)})
    groupedvalues=data.groupby('Gesture').sum().reset_index()

    b = sns.barplot(x = 'Gesture', y = 'Accuracy', data = groupedvalues)
    b.axes.set_title("Accuracy over different gestures",fontsize=20)
    b.set_xlabel('Gesture', fontsize=16)
    b.set_ylabel('Accuracy', fontsize=16)
    b.tick_params(labelsize=10)
    for index, row in groupedvalues.iterrows():
        b.text(row.name, row.Accuracy, round(row.Accuracy,2), color='black', ha="center", fontsize = 14)
    plt.tight_layout()
    
    savefig = input('Save figure? (y:1, n:0)')
    if savefig == 1:
        plt.savefig('CNN_accuracy_gestures.png')