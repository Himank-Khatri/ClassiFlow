import streamlit as st
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, average_precision_score, r2_score, mean_absolute_error, root_mean_squared_error, root_mean_squared_log_error, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.express as px



def run_models(X_train, X_test, y_train, y_test, models):
	
	def logistic_regression(params):
		return LogisticRegression(**params).fit(X_train, y_train).predict(X_test)
	
	def svc(params):
		return SVC(**params).fit(X_train, y_train).predict(X_test)
	
	def gaussian_nb(params):
		return GaussianNB(**params).fit(X_train, y_train).predict(X_test)
	
	def decision_tree(params):
		return DecisionTreeClassifier(**params).fit(X_train, y_train).predict(X_test)
	
	def random_forest(params):
		return RandomForestClassifier(**params).fit(X_train, y_train).predict(X_test)
	
	def knn(params):
		return KNeighborsClassifier(**params).fit(X_train, y_train).predict(X_test)

	col1, col2 = st.columns(2)

	for dic in models:
		params = dic['parameters']
		if dic['model'] == 'Logistic Regression':
			model = LogisticRegression(**params).fit(X_train, y_train)
		elif dic['model'] == 'Support Vector Machine':
			model = SVC(**params).fit(X_train, y_train)
		elif dic['model'] == 'Naive Bayes':
			model = GaussianNB(**params).fit(X_train, y_train)
		elif dic['model'] == "Decision Tree Classifier":
			model = DecisionTreeClassifier(**params).fit(X_train, y_train)
		elif dic['model'] == "Random Forest Classifier":
			model = RandomForestClassifier(**params).fit(X_train, y_train)
		elif dic['model'] == "K-Nearest Neighbors":
			model = KNeighborsClassifier(**params).fit(X_train, y_train)


		y_pred = model.predict(X_test)
		st.header("Results")		
		with st.container(border=True):
			st.subheader(dic['model'])
			col1, col2 = st.columns(2)
			with col1:
				st.dataframe(pd.DataFrame({"Metrics": ['Accuracy', 'Avg Precision', 'r2 Score', 'Mean Absolute Error', 'Root Mean Squared Error', 'Root Mean squared Log Error'], "Score": [accuracy_score(y_test, y_pred), average_precision_score(y_test, y_pred), r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), root_mean_squared_error(y_test, y_pred), root_mean_squared_log_error(y_test, y_pred)]}), hide_index=True)
			with col2:
				st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto = True, title='Confusion Matrix', height=400))
			# st.write(f"Accuracy: {}")
			# st.write(f"r2_score: {r2_score(y_test, y_pred)}")
			# st.write(f"mean_absolute_error: {mean_absolute_error(y_test, y_pred)}")
			# st.write(f"root_mean_squared_error: {root_mean_squared_error(y_test, y_pred)}")
			# st.write(f"root_mean_squared_log_error: {root_mean_squared_log_error(y_test, y_pred)}")
