import streamlit as st

st.cache_data
def input_logistic_regression():
    col1, col2 = st.columns(2)
    with col1:
        penalty = st.selectbox("penalty", options=['l2', 'l1', 'elasticnet', None])
        solver = st.selectbox('solver', options=['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
        fit_intercept = st.selectbox('fit_intercept', options=[True, False])
    with col2:
        C = st.number_input("C", value=1.0, step=0.1)
        max_iter = st.number_input("max_iter", value=100, step=50)
        tol = st.number_input('tol', value=0.001, step=0.0001, format='%0.7f')

    return {'penalty': penalty, 'C': C, 'solver': solver, 'max_iter': max_iter, 'fit_intercept': fit_intercept, 'tol':tol}

st.cache_data
def input_svm():
    col1, col2 = st.columns(2)
    with col1:
        kernel = st.selectbox("kernel", options=['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'])
        gamma = st.selectbox('gamma', options=['scale', 'auto'])
        tol = st.number_input('tol', value=0.001, step=0.0001, format='%0.7f')
    with col2:
        C = st.number_input("C", value=1.0, step=0.1)
        degree = st.number_input("degree (poly only)", value=3, step=1, min_value=1)

    return {'kernel': kernel, 'C': C, 'gamma': gamma, 'degree': degree, 'tol': tol}

st.cache_data
def input_dicision_tree():
    col1, col2 = st.columns(2)
    with col1:
        criterion = st.selectbox("criteria", options=['gini', 'entropy', 'log_loss'])
        max_features = st.selectbox("max_features", options=[None, 'sqrt', 'log2'])
    with col2:
        min_samples_split = st.number_input('min_samples_split', value=2.0, step=1.0)
        min_samples_leaf = st.number_input('min_samples_leaf', value=1, min_value=1, step=1)
    
    return {'criterion': criterion, 'max_features': max_features, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split}

st.cache_data
def input_random_forest():
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.number_input('n_estimators', value=100, step=1)
        criterion = st.selectbox("criteria", options=['gini', 'entropy', 'log_loss'])
        max_features = st.selectbox("max_features", options=[None, 'sqrt', 'log2'])
    with col2:
        min_samples_split = st.number_input('min_samples_split', value=2, step=1, min_value=2)
        min_samples_leaf = st.number_input('min_samples_leaf', value=1, min_value=1, step=1)
    
    return {'n_estimators': n_estimators, 'criterion': criterion, 'max_features': max_features, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split}


st.cache_data
def input_knn():
    col1, col2 = st.columns(2)
    with col1:
        weights = st.selectbox('weights', options=['uniform', 'distance'])
        algorithm = st.selectbox('algorithm', options=['auto', 'ball_tree', 'kd_tree', 'brute'])
        metric = st.selectbox('metric', options=['minkowski', 'euclidean', 'manhattan'])
    with col2:
        n_neighbors = st.number_input('n_neighbors', value=5, min_value=1, step=1)
        leaf_size = st.number_input('leaf_size', value=30, min_value=1, step=1)
        p = st.number_input('p', value=2, min_value=1, step=1)

    return {'weights': weights, 'algorithm': algorithm, 'metric': metric, 'n_neighbors': n_neighbors, 'leaf_size': leaf_size, 'p': p}

