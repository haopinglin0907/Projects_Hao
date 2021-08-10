from tqdm import tqdm
import pandas as pd
import glob2
import numpy as np


# split a univariate sequence into samples
def split_sequence(df, window_size, nonoverlap_size):
    '''Takes df of 8 channels of raw EMG data from Myo armband. window_size and nonoverlap_size indicating the rolling window paradigm'''
    X, y = list(), list()
    for i in np.arange(0, len(df), step = nonoverlap_size):
        # find the end of this pattern
        end_ix = i + window_size
        # check if we are beyond the sequence
        if end_ix > len(df)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = df.iloc[i:end_ix, :8], df.iloc[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def read_file(mode = 0, window_size = 52, nonoverlap_size = 5):
    '''
    mode: Integer indicating reading different data (0: PretrainData, 1: RetrainData, 2: TestData)
    window_size: Integer indicating the window_size to include in one sample
    nonoverlap_size: Integer indicating the size of the data in between samples, those data will be skipped
    '''
    X, y = list(), list()
    
    # Iterates through 4 cycles, each cycle consists of 8 gestures. In total 32 files of 5 seconds EMG readings
    for cycle in tqdm(range(1, 5)):
        if mode == 0:
            filename = glob2.glob('PretrainingData/Cycle{}/*'.format(cycle))
        elif mode == 1:
            filename = glob2.glob('EvaluationData/Test0/Cycle{}/*'.format(cycle))
        elif mode == 2:
            filename = glob2.glob('EvaluationData/Retraining/Cycle{}/*'.format(cycle))
        # 8 gestures from each cycle     
        for gesture in filename:
            col_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'Class']
            df = pd.read_csv(gesture, delimiter= ',', header = None, names = col_list)

            #Removes the '()' in C1, C8
            df['C1'] = pd.to_numeric(df['C1'].map(lambda x: x.replace('(', '')))
            df['C8'] = pd.to_numeric(df['C8'].map(lambda x: x.replace(')', '')))
            df['Class'] = df['Class'].astype('str')

            # Remove the 'Ready' class
            df['Class'] = df['Class'].str.replace(' ', '')
            df = df[(df['Class'] != 'Ready')]

            # Clean for the visual-motor delay (100 samples of begining of each gesture ==> 500 ms)
            for Class in df['Class'].unique():
                df_temp  = df.loc[df['Class'] == Class, ('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8')].copy()
                df_temp[:100] = np.nan
                df.loc[df['Class'] == Class, ('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8')] = df_temp
            df = df.dropna(axis = 0).reset_index(drop = True)

            X_temp, y_temp = split_sequence(df, window_size = 52, nonoverlap_size = 5)

            X.append(X_temp)
            y.append(y_temp)

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y