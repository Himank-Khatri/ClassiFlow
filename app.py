# streamlit run app.py --server.enableXsrfProtection false

import streamlit as st
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from visualization import *
from preprocessing import * 
from models import *
from inputs import *


st.set_page_config(layout="wide")
if 'models' not in st.session_state.keys():
	st.session_state['models'] = []
if 'last_added_model' not in st.session_state.keys():
	st.session_state['last_added_model'] = None
	

@st.dialog("Preprocess Data", width='large')
def preprocess_modal(dataset):
	columns = dataset.columns
	numeric_cols = list(dataset.select_dtypes(include=['int64', 'float64']).columns)
	
	dependent_variable = st.selectbox('Dependent Variable', options=columns, index=None, placeholder='Choose an option')
	
	col1, col2 = st.columns(2)
	
	with col1:
		null_handling_numeric = st.selectbox("Numeric column Null Handling", options=["Mean", "Median", "Mode", "Delete row"], index=None)
		null_handling_categorical = st.selectbox("Categorical column Null Handling", options=["Most Frequent", "Delete row"], index=None)
		one_hot_encoding_cols = st.multiselect("Apply One Hot Encoding", options=columns)
		normalize_cols = st.multiselect("Normalize Columns", options=numeric_cols)		
		test_size = st.slider("Test Size", 0.0, 1.0, 0.2, 0.01)
	with col2:
		delete_cols = st.multiselect("Remove Columns", options=columns)
		label_encoding_cols = st.multiselect("Apply Label Encoding", options=columns)
		min_max_cols = st.multiselect("Min-Max Scaling Columns", options=numeric_cols)		
		random_state = st.number_input("Random State", min_value=0, max_value=2**32-1, value=42)

	
	
	submit_button = st.button("Submit")

	
	if submit_button:
		if dependent_variable is None:
			st.info("No dependent variable chosen")
		else:
			st.session_state['preprocessing'] = {
				'delete_cols': delete_cols,
				'dependent_variable': dependent_variable,
				'null_handling_numeric': null_handling_numeric,
				'null_handling_categorical': null_handling_categorical,
				'label_encoding_cols': label_encoding_cols,
				'one_hot_encoding_cols': one_hot_encoding_cols,
				'normalize_cols': normalize_cols,
				'min_max_cols': min_max_cols,
				'test_size': test_size,
				'random_state': random_state,
			}
			st.rerun()
			
dataset = None

# data_tab, model_tab = st.tabs(["Data", "Model"])
data_tab, model_tab  = st.tabs(['Data', 'Model'])
# Sidebar
with st.sidebar:
	file = st.file_uploader('Import Dataset', type=['xlsx', 'csv'], help='Upload a dataset to work upon')
	if file:
		try:
			dataset = pd.read_csv(file, na_values=['', 'NaN',], keep_default_na=False)
		except:
			dataset = pd.read_excel(file, na_values=['', 'NaN'], keep_default_na=False)
		
		st.write("File loaded")

	preprocessing_button = st.button("Preprocess Data")
# Data tab
with data_tab:

	# Importing dataset
	if dataset is not None:
		st.dataframe(dataset)
	else:
		# dataset = pd.read_csv('sample.csv', na_values=['', 'NaN'], keep_default_na=False)
		dataset = pd.read_csv('datasets/titanic.csv', na_values=['', 'NaN'], keep_default_na=False)
		st.dataframe(dataset)
		# st.write("No dataset")

	# Ignore columns feature
	ignore_id = st.checkbox("Ignore Columns")
	if ignore_id:
		col1, _ = st.columns([1,1])
		with col1:
			ignored_columns = st.multiselect('Select columns to ignore', options=dataset.columns)
			old_dataset = dataset.copy()
			dataset = dataset.drop(ignored_columns, axis=1)
		
	# Visualizations
	col1, col2= st.columns([1,1], gap='medium'	)
	with col1:
		display_missing_values(dataset)
	with col2:
		display_categorical_distribution(dataset)
	display_numeric_distribution(dataset)
	display_scatter_matrix(dataset)
	# display_correlation_heatmap(dataset)


# st.session_state['preprocessing'] = {
# 	"delete_cols":['name'],
# 	"dependent_variable":"gender",
# 	"null_handling_numeric":"",
# 	"null_handling_categorical":'',
# 	"label_encoding_cols":['gender'],
# 	"one_hot_encoding_cols":['city'],
# 	"normalize_cols":['age'],
# 	"min_max_cols":['salary'],
# 	"test_size":0.33,
# 	"random_state":42
# }

st.session_state['preprocessing'] = {
	"delete_cols":['PassengerId', 'Name', 'Cabin', 'Ticket'],
	"dependent_variable":"Survived",
	"null_handling_numeric":"Mean",
	"null_handling_categorical":'Delete row',
	"label_encoding_cols":['Sex'],
	"one_hot_encoding_cols":['Embarked'],
	"normalize_cols":['Age'],
	"min_max_cols":['Fare'],
	"test_size":0.33,
	"random_state":42

}

@st.dialog("Choose a model", width="small")
def model_modal(container):
	model = st.selectbox("Select a model", options=["Logistic Regression", "Naive Bayes", "Support Vector Machine", "K-Nearest Neighbors", "Decision Tree Classifier", "Random Forest Classifier"], index=None)
	if model == 'Logistic Regression':
		params = input_logistic_regression()
	elif model == 'Support Vector Machine':
		params = input_svm()
	elif model == 'Naive Bayes':
		params = {}	
	elif model == 'Decision Tree Classifier':
		params = input_dicision_tree()
	elif model == 'Random Forest Classifier':
		params = input_random_forest()
	elif model == 'K-Nearest Neighbors':
		params = input_knn()	
	

	if st.button("Add"):
		st.session_state['last_added_model'] = {'model': model, 'parameters': params}
		st.session_state['models'].append(st.session_state['last_added_model'])
		st.rerun()


if preprocessing_button:
	preprocess_modal(dataset)


if 'preprocessing' in st.session_state.keys():
	with data_tab:	
		st.success("Data preprocessed. Head over to train model tab to train the model.")
	with model_tab:
		try:
			X_train, X_test, y_train, y_test = preprocess(dataset, st.session_state['preprocessing'])
			preprocess_error = False
		except Exception as e:
			st.error("Error in data preprocessing. Please try again with proper methods.")
			st.error(e)
			preprocess_error = True
			
		if not preprocess_error:
			with st.expander("Train Test Split"):
				col1, col2 = st.columns([5,2])
				with col1:
					st.write("X train")
					st.dataframe(X_train, height=200, use_container_width=True)
					st.write("X test")
					st.dataframe(X_test, height=200, use_container_width=True)
				with col2:
					st.write("y train")
					st.dataframe(y_train, height=200)
					st.write("y test")
					st.dataframe(y_test, height=200)

				export_data = st.download_button(label="Download", data=X_train.to_csv(index=False), file_name='preprocessed.csv') 
			
			model_container = st.container()

			st.session_state['models'] = [{"model":"Logistic Regression","parameters":{"penalty":"l2","C":1,"solver":"lbfgs","max_iter":100,"fit_intercept":True,"tol":0.001}},{"model":"Naive Bayes","parameters":{}},{"model":"Support Vector Machine","parameters":{"kernel":"rbf","C":1,"gamma":"scale","degree":3,"tol":0.001}},{"model":"Decision Tree Classifier","parameters":{"criterion":"gini","max_features":None,"min_samples_leaf":1,"min_samples_split":2}}]
									
			if st.button("Add Model"):
				model_modal(model_container)

			if len(st.session_state['models']) > 0:
				for i in st.session_state['models']:
					model_container.subheader(i['model'])
					model_container.caption(str(i['parameters']))
			else:
				model_container.subheader("Add models to show")

			if st.button("Run Models"):
				run_models(X_train, X_test, y_train, y_test, st.session_state['models'])

				# for model, y_pred in predictions.items():
			