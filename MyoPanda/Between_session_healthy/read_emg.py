from tqdm import tqdm
import pandas as pd
import glob2
import numpy as np
import json
from scipy import signal

def butter_highpass(cutoff, fs, order=3):
    nyq = .5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=3):
    b, a = butter_highpass(cutoff=cutoff, fs=fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


# split a univariate sequence into samples
def split_sequence(df, window_size, nonoverlap_size):
    '''Takes df of 8 channels of raw EMG data from Myo armband. window_size and nonoverlap_size indicating the rolling window paradigm'''
    X, y, ID = list(), list(), list()
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
        ID.append(df.iloc[i, -3])
    return np.array(X), np.array(y), np.array(ID)


def preprocess(df, window_size = 52, nonoverlap_size = 5, highpass = True):
    '''
    window_size: Integer indicating the window_size to include in one sample
    nonoverlap_size: Integer indicating the size of the data in between samples, those data will be skipped
    '''

    X, y, ID = list(), list(), list()
    subject_list = df['ID'].unique()
    gesture_list = df['Gesture'].unique()
    session_list = df['session'].unique()

    for subject in subject_list:  
        for session in session_list:
#             print(f'Processing subject: {subject}\n')
            for gesture in gesture_list:
                print(f'\nGesture: {gesture}')

                trial_list = df['Trial_num'].unique()

                for trial_num in trial_list:
#                     print(f'Trial: {trial_num}')
                    df_temp = df.copy()
                    df_temp = df_temp.loc[(df_temp['ID'] == subject) & 
                                          (df_temp['Gesture'] == gesture) & 
                                          (df_temp['Trial_num'] == trial_num)&
                                          (df_temp['session'] == session)]

                    if len(df_temp) == 0:
                        continue

                    X_temp, y_temp, ID_temp = split_sequence(df_temp, window_size = 52, nonoverlap_size = 5)
                    X.append(X_temp)
                    y.append(y_temp)
                    ID.append(ID_temp)

    X = np.vstack(X)
    y = np.hstack(y)
    ID = np.hstack(ID)
    if highpass == True:
        # high pass filtering
        for sample in range(0, len(X)):
            for channel in range(0, 8):
                X[sample, :, channel] = butter_highpass_filter(X[sample, :, channel], 2, 200, order = 3)
                
    return X, y, ID


def read_file():
    '''
    
    '''
    df = pd.DataFrame(None)
    jsonfile_path = glob2.glob('**/**.json')
    for jsonfile in tqdm(jsonfile_path):
        gesture = jsonfile.split('-')[1]
        trial_num = jsonfile.split('-')[2]
        site = jsonfile.split('-')[3]
        session = jsonfile.split('_')[1][:2]
        subject = jsonfile.split('_')[0]
        #print(f'Processing subject: {subject}')
        f = open(jsonfile)
        data = json.load(f)
        
        data['emgData']['Trial_num'] = trial_num 
        data['emgData']['Site'] = site
        data['emgData']['ID'] = subject
        data['emgData']['session'] = session
        data['emgData']['Gesture'] = gesture
        data = pd.DataFrame(data['emgData'])
        df = pd.concat((df, data), axis = 0)
    return df