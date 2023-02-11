'''This module is used to process the data ready to be  inputed into the NN'''

import logging
import pandas as pd
import numpy as np

FILE_LOCATION = 'heart.dat'


def read_data(f_l=FILE_LOCATION):
    '''This function takes in the location of the data, and opens the data
    labeling all the columns. It also checks that there is no missing data
    and that all data is numeric as needed for a NN'''

    headers =  ['age', 'sex','chest_pain_type','resting_blood_pressure',
        'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
        'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',"slope_of_peak",
        'num_of_major_vessels','thal', 'heart_disease']


    df = pd.read_csv(f_l, sep=' ', names=headers)

    #checks if there are any null values in data
    if df.isnull().sum().any():
        logging.error('Missing some values in data')


    if not df.shape[1] == df.select_dtypes(include=np.number).shape[1]:
        print('dgf')
    return df


read_data()