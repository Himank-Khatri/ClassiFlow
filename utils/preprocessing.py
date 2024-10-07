import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def preprocess(dataset, preprocessing):
    
    if preprocessing['delete_cols']:
        dataset = dataset.drop(preprocessing['delete_cols'], axis=1)

    dataset = handle_null_values(dataset, preprocessing)
    dataset = encode(dataset, preprocessing)
    # st.write(dataset.columns)
    X = dataset.drop(preprocessing['dependent_variable'], axis=1)
    y = dataset[preprocessing['dependent_variable']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=preprocessing['test_size'], random_state=preprocessing['random_state'])

    X_train, X_test = scale(X_train, X_test, preprocessing)

    return X_train, X_test, y_train, y_test




def handle_null_values(dataset, preprocessing):
    null_handling_numeric, null_handling_categorical = preprocessing['null_handling_numeric'], preprocessing['null_handling_categorical']
    

    if null_handling_categorical == 'Delete row' and null_handling_numeric == 'Delete row':
        dataset = dataset.dropna(axis=0)
    
    else:
        categorical_cols = dataset.select_dtypes(include=[object]).columns
        numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns  
    
        if null_handling_numeric == 'Delete row' and null_handling_categorical != 'Delete row':
            dataset = dataset.dropna(subset=numeric_cols, axis=0)
            # dataset[categorical_cols] = SimpleImputer(strategy='most_frequent', ).fit_transform(dataset[categorical_cols])
            dataset[categorical_cols] = dataset[categorical_cols].apply(lambda column: column.fillna(column.mode()[0]))

        elif null_handling_categorical == 'Delete row' and null_handling_numeric != 'Delete row':
            dataset = dataset.dropna(subset=categorical_cols, axis=0)
            if null_handling_numeric.lower() == 'mean':
                dataset[numeric_cols] = dataset[numeric_cols].apply(lambda column: column.fillna(column.mean()))
            elif null_handling_numeric.lower() == 'median':
                dataset[numeric_cols] = dataset[numeric_cols].apply(lambda column: column.fillna(column.median()))
            else:
                dataset[numeric_cols] = dataset[numeric_cols].apply(lambda column: column.fillna(column.mode()[0]))
        
        else:
            dataset[numeric_cols] = dataset[numeric_cols].apply(lambda column: column.fillna(column.mean()))    
            dataset[categorical_cols] = dataset[categorical_cols].apply(lambda column: column.fillna(column.mode()[0]))
    
    return dataset

def encode(dataset, preprocessing):

    label_columns = preprocessing['label_encoding_cols']
    oh_columns = preprocessing['one_hot_encoding_cols']

    for column in label_columns:
        dataset[column] = LabelEncoder().fit_transform(dataset[column])
    if oh_columns:
        dataset = pd.get_dummies(dataset, columns=oh_columns, drop_first=False, dtype='int8')


    return dataset


def scale(X_train, X_test, preprocessing):
    if preprocessing['normalize_cols']:
        sr = StandardScaler()
        X_train[preprocessing['normalize_cols']] = sr.fit_transform(X_train[preprocessing['normalize_cols']])
        X_test[preprocessing['normalize_cols']] = sr.transform(X_test[preprocessing['normalize_cols']])

    if preprocessing['min_max_cols']:
        mms = MinMaxScaler()
        X_train[preprocessing['min_max_cols']] = mms.fit_transform(X_train[preprocessing['min_max_cols']])
        X_test[preprocessing['min_max_cols']] = mms.transform(X_test[preprocessing['min_max_cols']])

    return X_train, X_test