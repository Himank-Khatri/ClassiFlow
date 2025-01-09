import streamlit as st
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, average_precision_score, r2_score, mean_absolute_error, root_mean_squared_error, root_mean_squared_log_error, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go



def run_models(X_train, X_test, y_train, y_test, models):

	if len(models) > 0:
		st.header("Results")		

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
		with st.container(border=False):
			col1, col2, col3 = st.columns(3)
			with col1:
				st.subheader(dic['model'])
				st.caption(params)
				st.dataframe(pd.DataFrame({"Metrics": ['Accuracy', 'Avg Precision', 'r2 Score', 'Mean Absolute Error', 'Root Mean Squared Error', 'Root Mean squared Log Error'], "Score": [accuracy_score(y_test, y_pred), average_precision_score(y_test, y_pred), r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), root_mean_squared_error(y_test, y_pred), root_mean_squared_log_error(y_test, y_pred)]}), hide_index=True)
			with col2:
				st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto = True, title='Confusion Matrix', height=350))
			with col3:
				fpr, tpr, _ = roc_curve(y_test, y_pred) 
				roc_auc = auc(fpr, tpr)

				fig_roc = go.Figure()
				fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})', line=dict(color='powderblue')))
				fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='orangered', dash='dash')))

				fig_roc.update_layout(
					title="Receiver Operating Characteristic (ROC)",
					xaxis_title="False Positive Rate",
					yaxis_title="True Positive Rate",
					showlegend=True,
					xaxis=dict(range=[0, 1]),
					yaxis=dict(range=[0, 1])
				)
				fig_roc.update_layout(
					legend=dict(
						x=0.5,  # x position (horizontal)
						y=0.2,  # y position (vertical)
						traceorder='normal',
						xanchor='center',
						yanchor='top'
					)
				)
				st.plotly_chart(fig_roc, key=None)
