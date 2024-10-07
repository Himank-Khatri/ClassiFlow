# Classification Builder

**Classification Builder** is a user-friendly web application built with Streamlit that allows users to easily perform exploratory data analysis (EDA), preprocess datasets, train machine learning classification models, and evaluate them using various performance metrics and visualizations, such as ROC curves and confusion matrices.

## Features

- **Data Import**: Upload datasets in `.csv` or `.xlsx` format.
- **Preprocessing**:
  - Handle missing values for both numeric and categorical columns.
  - Apply one-hot encoding, label encoding, normalization, and scaling techniques.
  - Remove unwanted columns and customize test/train splits.
- **Exploratory Data Analysis (EDA)**:
  - View summary statistics of the dataset.
  - Visualize missing values, categorical distributions, numeric distributions, and scatter matrices.
- **Model Training**: Choose from several built-in classification algorithms:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier
  - Random Forest Classifier
- **Model Evaluation**:
  - Train-test split visualization.
  - ROC curve, confusion matrix, and various model performance metrics.
- **Model Comparison**: Add and compare multiple models on the same dataset.

## Installation

To run this app locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/Himank-Khatri/classification-builder.git
    ```

2. Navigate to the project directory:

    ```bash
    cd classification-builder
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run app.py --server.enableXsrfProtection false
    ```

## Demo

Check out the live web app here: [Classification Builder](https://classification-builder.streamlit.app/)

## Project Structure

- `app.py`: The main Streamlit application file.
- `utils/`: Contains the helper functions for preprocessing, model training, and visualization.
- `requirements.txt`: List of dependencies required to run the app.

## Contributing

Feel free to contribute to the project by creating issues or submitting pull requests on the GitHub repository: [GitHub Repository](https://github.com/Himank-Khatri/classification-builder/).

## License

This project is licensed under the MIT License.
