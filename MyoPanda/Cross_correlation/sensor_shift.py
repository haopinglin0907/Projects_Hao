import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_custom_palette(series, max_color = 'red', other_color = 'orange'):
    max_val = series.max()
    pal = []
    
    for item in series:
        if item == max_val:
            pal.append(max_color)
        else:
            pal.append(other_color)
    return pal


### Calculating the similarity using the channel means
def cross_correlation_vector(A, B):
    A_vector = []
    B_vector = []
    for channel in range(0, A.shape[1]):
        A_temp = A[:, channel] 
        A_channel_mean = A_temp.mean()
        A_vector.append(A_channel_mean)
        
        B_temp = B[:, channel]
        B_channel_mean = B_temp.mean()
        B_vector.append(B_channel_mean)
        
    A_vector = np.array(A_vector)
    B_vector = np.array(B_vector)
    C = np.multiply(A_vector, B_vector).sum() / (np.sqrt((A_vector**2).sum() * (B_vector**2).sum()))

    return C

def CC_to_shift(df, subject_reference, subject_to_rotate, calibration_gesture_list, session_reference, session_to_rotate, ax = None):
#     print(f'Current Subject: {subject_to_rotate}')
    shift_range = range(0, 8)

    max_CC = 0
    final_status = False
    final_shift = 0

    ## Reference session will not get flipped, but the test session
    CC_list = []
    for flipped in [False, True]:

        for shift in shift_range:
            A_vector_all_gesture = np.zeros(shape = (1, len(calibration_gesture_list) * 8))
            B_vector_all_gesture = np.zeros(shape = (1, len(calibration_gesture_list) * 8))
#             print(f'Shift: {shift}')

            for calibration_gesture in calibration_gesture_list:
                df_A = df[(df['ID'] == subject_reference) & 
                          (df['Gesture'] == calibration_gesture) & 
                          (df['Trial_num'] <= 3) & 
                          (df['session'] == session_reference)].iloc[:, :8]            
                df_A = df_A[df_A.index > 100]
                A_value = df_A.iloc[:, :8].rolling(1).mean().dropna().values

                ## concatenation of all gestures for sessionA
                ## Check if it is all zeros, if so, replace that with the first gesture
                ## If not, concatenate the 2nd gesture with the first one and so on
                if not np.any(A_vector_all_gesture):
                    A_vector_all_gesture = A_value
                else:
                    length = np.minimum(len(A_vector_all_gesture), len(A_value))
                    A_vector_all_gesture = np.concatenate((A_vector_all_gesture[:length], A_value[:length]), axis = 1)

                if flipped == False:    

                    df_B = df[(df['ID'] == subject_to_rotate) & 
                              (df['Gesture'] == calibration_gesture) & 
                              (df['Trial_num'] <= 3) & 
                              (df['session'] == session_to_rotate)].iloc[:, :8]
                else:
    #                 print('Flipping the sensor\n')
                    df_B = flipping_sensor(df[(df['ID'] == subject_to_rotate)  & 
                                              (df['Gesture'] == calibration_gesture) & 
                                              (df['Trial_num'] <= 3) & 
                                              (df['session'] == session_to_rotate)]).iloc[:, :8]

                df_B = df_B[df_B.index > 100]
                B = df_B.iloc[:, :8].rolling(1).mean().dropna()     

                B_value = pd.concat((B,B), axis = 1).iloc[:,  0 + shift:8 + shift].values

                ## concatenation of all gestures for sessionB (shift indiviually and concatenate)
                ## Check if it is all zeros, if so, replace that with the first gesture
                ## If not, concatenate the 2nd gesture with the first one and so on
                if not np.any(B_vector_all_gesture):
                    B_vector_all_gesture = B_value
                else:
                    length = np.minimum(len(B_vector_all_gesture), len(B_value))
                    B_vector_all_gesture = np.concatenate((B_vector_all_gesture[:length], B_value[:length]), axis = 1)

            C = cross_correlation_vector(A_vector_all_gesture, B_vector_all_gesture)
#             print(f'CC: {C:.4f}\n')
            CC_list.append(C)

    try:
        shift_max = np.where(CC_list == np.max(CC_list))[0][0]
        palette = set_custom_palette(pd.Series(CC_list))

        if shift_max < 8:
            final_status = False
            final_shift = shift_max
        else:
            final_status = True
            final_shift = shift_max - 8
            
        
        ax.bar(x = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7',
                           'F_S0', 'F_S1', 'F_S2', 'F_S3', 'F_S4', 'F_S5', 'F_S6', 'F_S7'], 
                       height = CC_list, alpha = 0.7, color = palette)
        ax.set_title(subject_to_rotate)
        ax.set_ylabel('Cross-correlation')
        ax.set_xlabel('Potential shifts')
        plt.setp(ax.get_xticklabels(), rotation=45)

    except Exception as e:
        print(e)

#     print(f'Flipping needed? {final_status}\n')
    return final_status, final_shift


def shift_to_rotation(df, shift, subject_reference, subject_to_rotate, session_reference, session_to_rotate, flipped = False):
    df_train = df[(df['ID'] == subject_reference) & 
                  (df['session'] == session_reference) & (df['Trial_num'] <=3) & 
                  (df['Gesture'] != '1') & (df['Gesture'] != '0') & (df['Gesture'] != '3') & 
                  (df['Gesture'] != '10') & (df['Gesture'] != '11')]
        
    df_test = df[(df['ID'] == subject_to_rotate) & 
                 (df['session'] == session_to_rotate) & (df['Trial_num'] <=3) & 
                 (df['Gesture'] != '1') & (df['Gesture'] != '0') & (df['Gesture'] != '3') & 
                 (df['Gesture'] != '10') & (df['Gesture'] != '11')]
    
    if flipped == True:
        print('Flipping data')
        df_test_temp = flipping_sensor(df_test.iloc[:, :8])
    else:
        df_test_temp = df_test.iloc[:, :8].copy()
    
    df_test_calibrated = pd.concat((pd.concat((df_test_temp, df_test_temp), axis = 1).iloc[:,  0 + shift:8 + shift], df_test.iloc[:, 8:]), axis = 1)
    
    df_test_calibrated = pd.DataFrame(df_test_calibrated.values, columns = df_test.columns)
            
    return df_train, df_test, df_test_calibrated


def flipping_sensor(df_unflipped):
    A = df_unflipped.iloc[:, :8]
    A = A.loc[:, ::-1].values
    
    df_flipped = df_unflipped.copy()
    df_flipped.iloc[:, :8] = A
    
    return df_flipped