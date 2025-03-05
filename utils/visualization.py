import pandas as pd
import streamlit as st
import plotly.express as px
import plost

def display_missing_values(dataset):
    missing_values = pd.DataFrame({'Null Values': dataset.isnull().sum()})
    missing_values['Non-Null Values'] = len(dataset) - missing_values['Null Values']
    missing_values['columns'] = missing_values.index
    plost.bar_chart(missing_values, value=['Non-Null Values', 'Null Values'], bar='columns', title='Null Values', direction='horizontal', stack='center')

    # missing_cols = missing_values[missing_values['Null Values']>0]
    # if missing_cols.empty:
    # 	st.write("No null values")
    # else:
    # 	missing_cols['column'] = missing_cols.index
    # 	plost.pie_chart(data=missing_cols, theta='Null Values', color="column", title="Null Distribution", legend='left')

def display_numeric_distribution(dataset):
    numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
    numeric_df = dataset[numeric_cols]
    fig = px.box(numeric_df, title='Numeric Data Distribution')
    st.plotly_chart(fig, key=f'{fig}')
    
def display_categorical_distribution(dataset):
    categorical_cols = dataset.select_dtypes(include=object).columns
    categorical_df = dataset[categorical_cols].nunique(dropna=False).reset_index()
    categorical_df.columns = ['column', 'count']
    plost.bar_chart(categorical_df, value='count', bar='column', title='Categorical Distribution', direction='verticle', use_container_width=True)
        
    
def display_correlation_heatmap(dataset):
    numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
    numeric_df = dataset[numeric_cols]
    correlation_matrix = numeric_df.corr()
    fig = px.imshow(correlation_matrix, text_auto=True, title="Correlation Heatmap", height=700)
    fig.update_yaxes(showticklabels=True, tickangle=-45)
    fig.update_xaxes(showticklabels=True, tickangle=315)
    st.plotly_chart(fig, key=f'{fig}')

def display_scatter_matrix(dataset):
    numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
    numeric_df = dataset[numeric_cols]
    fig = px.scatter_matrix(numeric_df, title="Scatter Matrix")
    st.plotly_chart(fig, key=f'{fig}')

    
