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

    #convert data set boolean to conventional 0, 1
    df['heart_disease'].replace({1: 0, 2: 1}, inplace=True)

    #checks if there are any null values in data
    if df.isnull().sum().any():
        logging.error('Missing some values in data')

    #checks if all data is numeric
    if not df.shape[1] == df.select_dtypes(include=np.number).shape[1]:
        logging.error('Some data is not numeric')

    return df


def train_test_split(df, test_size=0.2, shuffle=True, ):
    '''This takes in a data set, and returns it split into training 
    and testing data aswell as feature and target split.

    test_size - the perecent of data to be used for testing
    shuffle - whether or not to shuffle data before splitting'''
    if test_size > 1 or test_size < 0:
        logging.error('Test size is not valid')

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    num_row_test = round(len(df)*test_size)

    test_data = df.iloc[:num_row_test,:].reset_index(drop=True)
    train_data = df.iloc[num_row_test:,:].reset_index(drop=True)

    x_test, y_test = split_data(test_data)
    x_train, y_train = split_data(train_data)

    return(x_test, y_test, x_train, y_train)


def split_data(df):
    '''This takes in a data frame and splits the target from the feature
    returning two variables'''

    x_data = df.drop(columns=['heart_disease'])

    y_data = df['heart_disease']

    return(x_data, y_data)


class Standardizer:
    '''This class is used to normalize the data'''
    mean = None
    std = None

    def fit(self, df):
        '''this method finds the mean and std of each column given'''
        self.mean = df.mean()
        self.std = df. std()

    def transform(self, df):
        '''This normalizes the inputed data using the already defined
        mean and std'''

        norm_df = (df-self.mean)/self.std

        return norm_df


def main():

    df = read_data()
    x_test, y_test, x_train, y_train = train_test_split(df)

    stdzr = Standardizer()
    stdzr.fit(x_train)
    x_train = stdzr.transform(x_train)
    x_test = stdzr.transform(x_test)

    return(x_test, y_test, x_train, y_train)

main()