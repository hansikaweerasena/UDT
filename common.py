# Load packages we need
import sys
import os
import time

import numpy as np
import pandas as pd
import sklearn
import seaborn as sns

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.utils import resample
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, auc


from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 14})


def var_exists(var_name):
    return (var_name in globals() or var_name in locals())


def load_data(path='diabetes_012_health_indicators_BRFSS2015.csv'):
    diabetes_data = pd.read_csv(path)
    print(f'Data loaded successfully data shape:', diabetes_data.shape )
    return diabetes_data


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f'Seed set to {seed}')


def plot_multiclass_roc_curve(y, prob_pred_y, classes, title, file_name='ROC_Curve_NN_UP.jpg'):
    plt.figure(figsize=(8, 6))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y == i, prob_pred_y[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name, dpi=400)
    plt.show()


def split_and_scale_data(data_df,  scaler='MinMax', oneHotEncode=False, random_state=42, target='Diabetes_012', test_val_prop=0.1, val_prop=0.5, verbose=True):
   
    # Split the data into features and labels
    x_all = data_df.drop('Diabetes_012', axis=1)

    feature_columns = x_all.columns
    y_all = data_df['Diabetes_012']

    if oneHotEncode:
        num_classes = 3
        y_all = keras.utils.to_categorical(y_all, num_classes)

    # Split the data into training, validation, and test sets
    train_x, temp_x, train_y, temp_y = train_test_split(x_all, y_all, test_size=test_val_prop, random_state=random_state)
    val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=val_prop, random_state=random_state)

    if verbose: 
        print('Shape : train x: {} y: {}, val x: {} y: {}, test shape: {} y: {}'.format(train_x.shape, train_y.shape,
                                                                                      val_x.shape, val_y.shape, test_x.shape, test_y.shape))

    if scaler == 'MinMax':
        scaler = MinMaxScaler()
    elif scaler == 'Standard':
        scaler = StandardScaler()
    else:
        raise ValueError('Invalid scaler')

    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)

    return train_x, train_y, val_x, val_y, test_x, test_y, feature_columns


def resample_data(train_x, train_y, feature_columns, resample_type='UP', target='Diabetes_012', verbose=False):

    train_df = pd.DataFrame(train_x, columns=feature_columns)

    is_one_hot_encoded = False

    if train_y.shape[1] > 1:
        is_one_hot_encoded = True
        train_y = np.argmax(train_y, axis=1)

    train_df[target] = train_y

    if verbose:
        train_df.head()
        sns.countplot(x='Diabetes_012', data=train_df)
        plt.xlabel('No Diabetes, pre-diabetes, or diabetes')
        plt.ylabel('Count')
        plt.title('Distribution of target variable')

        legend_labels = ['No Diabetes', 'Pre-diabetes', 'Diabetes']
        legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in sns.color_palette()[:3]]  
        plt.legend(legend_handles, legend_labels)

        plt.show()

    if resample_type == 'SMOTE':
        print('Resampling using SMOTE')
        train_x, train_y = smote(train_x, train_y, verbose=verbose)
    elif resample_type == 'UP':
        print('Resampling using Upsampling')
        train_df = upsample(train_df, verbose=verbose)
        train_x = train_df.drop(target, axis=1).values
        train_y = train_df[target]
    elif resample_type == 'DOWN':
        print('Resampling using Downsampling')
        train_df = downsample(train_df, verbose=verbose)
        train_x = train_df.drop(target, axis=1).values
        train_y = train_df[target]
    
    if is_one_hot_encoded:
        num_classes = 3
        train_y = keras.utils.to_categorical(train_y, num_classes)

    return train_x, train_y
    

def upsample(train_df, verbose=False, random_state=42):
    # # #1: Upsampling (Over-sampling)
    # Separate majority and minority classes
    majority_class = train_df[train_df['Diabetes_012'] == 0]
    minority_class1 = train_df[train_df['Diabetes_012'] == 1]
    minority_class2 = train_df[train_df['Diabetes_012'] == 2]

    if verbose:
        print(f'Original class distribution: {len(majority_class)} {len(minority_class1)} {len(minority_class2)}')

    # Upsample minority class
    minority_upsampled1 = resample(minority_class1, 
                                replace=True,     # sample with replacement
                                n_samples=len(majority_class),    # to match majority class
                                random_state=random_state)

    minority_upsampled2 = resample(minority_class2, 
                                replace=True,     # sample with replacement
                                n_samples=len(majority_class),    # to match majority class
                                random_state=random_state)

    # Combine majority class with upsampled minority class
    upsampled = pd.concat([majority_class, minority_upsampled1, minority_upsampled2])

    # Shuffle the upsampled DataFrame to ensure randomness
    upsampled = upsampled.sample(frac=1, random_state=42).reset_index(drop=True)

    target_distribution = upsampled['Diabetes_012'].value_counts()
    if verbose:
        print(target_distribution)
    return upsampled


def downsample(train_df, verbose=False, random_state=42):
    # # #2: Downsampling (Under-sampling)
    majority_class = train_df[train_df['Diabetes_012'] == 0]
    minority_class1 = train_df[train_df['Diabetes_012'] == 1]
    minority_class2 = train_df[train_df['Diabetes_012'] == 2]

    if verbose:
         print(f'Original class distribution: {len(majority_class)} {len(minority_class1)} {len(minority_class2)}')

    # Downsample majority class
    majority_downsampled1 = resample(majority_class, 
                                    replace=False,    # sample without replacement
                                    n_samples=len(minority_class1),
                                    random_state=random_state)

    majority_downsampled2 = resample(minority_class2, 
                                    replace=False,
                                    n_samples=len(minority_class1), 
                                    random_state=random_state)

    # Combine minority class with downsampled majority classes
    downsampled = pd.concat([majority_downsampled1, minority_class1, majority_downsampled2])

    # Shuffle the downsampled DataFrame to ensure randomness
    downsampled = downsampled.sample(frac=1, random_state=42).reset_index(drop=True)

    target_distribution = downsampled['Diabetes_012'].value_counts()
    if verbose:
        print(target_distribution)
    return downsampled


def smote(train_x, train_y, verbose=False, random_state=42):
    from imblearn.over_sampling import SMOTE
    # # #3: Apply SMOTE: generates synthetic samples for the minority class to balance the class distribution

    smote = SMOTE(random_state=random_state)
    X_smote, y_smote = smote.fit_resample(train_x, train_y)

    return X_smote, y_smote
