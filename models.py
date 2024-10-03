import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix


def run_models(X_train, X_test, y_train, y_test, models):
	
	def logistic_regression(params):
		return LogisticRegression(**params).fit(X_train, y_train).predict(X_test)
	
	def svm(params):
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
		if dic['model'] == 'Logistic Regression':
			y_pred = logistic_regression(dic['parameters'])
		elif dic['model'] == 'Support Vector Machine':
			y_pred = svm(dic['parameters'])
		elif dic['model'] == 'Naive Bayes':
			y_pred = gaussian_nb(dic['parameters'])
		elif dic['model'] == "Decision Tree Classifier":
			y_pred = decision_tree(dic['parameters'])
		elif dic['model'] == "Random Forest Classifier":
			y_pred = random_forest(dic['parameters'])
		elif dic['model'] == "K-Nearest Neighbors":
			y_pred = knn(dic['parameters'])

		with st.container():
			st.subheader(dic['model'])
			st.write(confusion_matrix(y_test, y_pred))
			st.write(accuracy_score(y_test, y_pred))
