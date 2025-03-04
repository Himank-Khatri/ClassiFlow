# ClassiFlow

**Classification Builder** is a user-friendly web application built with Streamlit that allows users to easily perform exploratory data analysis (EDA), preprocess datasets, train machine learning classification models, and evaluate them using various performance metrics and visualizations.

## Demo

Check out the live web app here: [ClassiFlow](https://classiflow.streamlit.app/)


## Features

- **Data Import**: Upload `.csv` or `.xlsx` datasets.
- **Preprocessing**: Handle missing values, encode categorical data, normalize, scale, and split datasets for training/testing.
- **EDA**: Visualize missing values, data distributions, and scatter plots.
- **Model Training**: Choose from Logistic Regression, Naive Bayes, SVM, KNN, Decision Tree, and Random Forest classifiers.
- **Evaluation**: Visualize performance metrics such as ROC curves, confusion matrices, and accuracy.
- **Model Comparison**: Add and compare multiple models.

## Installation

To run this app locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/Himank-Khatri/classiflow.git
    ```

2. Navigate to the project directory:

    ```bash
    cd classiflow
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run app.py --server.enableXsrfProtection false
    ```

## Project Structure

- `app.py`: The main Streamlit application file.
- `utils/`: Contains the helper functions for preprocessing, model training, and visualization.
- `requirements.txt`: List of dependencies required to run the app.

## Contributing

Feel free to contribute to the project by creating issues or submitting pull requests on the GitHub repository: [GitHub Repository](https://github.com/Himank-Khatri/ClassiFlow/).

## License

This project is licensed under the MIT License.
