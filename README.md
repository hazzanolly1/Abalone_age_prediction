## Table of contents
1. Introduction
2. Dataset
3. Features
4. Regression Models
5. Usage 
6. Dependencies
7. Installation  
8. References
9. License

## 1.0 Introduction
Abalone Dataset Regression Models:
This program implements several regression models to predict the number of rings in abalone based on various physical measurements. Abalone is a type of marine mollusc, and the number of rings is an indicator of its age.
## 2.0 Dataset
The dataset "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data" used in this program contains information about abalone, including physical measurements such as length, diameter, height, and weights of different parts. The target variable is the number of rings, which serves as a proxy for the age of the abalone.

## 3.0 Features
1. Sex: Categorical variable indicating the gender of the abalone (one-hot encoded)
2. Length: Length of the abalone in millimeters
3. Diameter: Diameter of the abalone in millimeters
4. Height: Height of the abalone in millimeters
5. WholeWeight: Weight of the whole abalone in grams
6. ShuckedWeight: Weight of the abalone's meat in grams
7. VisceraWeight: Weight of the abalone's gut in grams
8. ShellWeight: Weight of the abalone's shell in grams
## Regression Models
The following regression models are implemented in the program:
1. Linear Regression:
Used to predict value of a variable(dependent variable) using another variable(independent variable)It fits a linear model to the relationship between the input features and the target variable. It finds the best-fitting line through the data points.
2. LASSO(Least Absolute Shrinkage and Selection Operator) Regression: 
Adds an L1 penalty term to the loss function,performs feature selection by shrinking some coefficients to zero, thus providing a simpler model and potentially improving generalization performance
Why it was utilized: Lasso regression .
3. Ridge Regression:
Adds an L2 penalty term to the loss function, which penalizes large coefficients(never attains zero but small value) and helps prevent overfitting.It is useful when the dataset contains multi-collinearity or when regularization is necessary.
4. Bagging (Bootstrap Aggregating)Regression:
Involves training multiple models on random subsets of the training data and then averaging their predictions.Bagging reduces variance and helps improve model stability and robustness. It is useful for reducing the impact of overfitting and improving model generalization.
5. Random Forest: 
Is an ensemble learning method that builds multiple decision trees during training and aggregates their predictions to make a final prediction.It provides higher predictive accuracy compared to individual decision trees. It handles non-linear relationships well and is robust to overfitting.
6. LightGBM:
Is a gradient boosting framework that uses a tree-based learning algorithm. It grows trees vertically (leaf-wise) and minimizes the loss function. It is known for its high performance, efficiency, and scalability. It handles large datasets well and provides excellent predictive accuracy with minimal tuning.

## 4.0 Usage
The Abalone Regression program involves underlisted tasks: 

1. Data Preparation
a) Loading the Dataset:
Abalone dataset was loaded into the Python environment from "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data".
b) Data Cleaning (Handling missing values):
A check for missing values was initially done. and subsequent filling missing values with the mean or median or removal of rows or columns with missing values in the dataset. 
c) Feature Engineering (Encoding Categorical Variable):The feature engineering process involves using One-hot encoding to convert the categorical variable "Sex" into a numerical format.The "Sex" column in the dataset is a categorical variable, meaning it contains non-numeric values.This involves creating dummy variables for each category of the "Sex" column.
The pd.get_dummies function from the Pandas library is used for subject one-hot encoding.This ensures that the resulting dataset consists of features (X) and the target variable (y) which are numerical input data suitable for training and evaluating machine learning models as most machine learning algorithms require numerical inputs. 
The drop_first=True parameter is set to drop the first dummy variable to avoid multi-collinearity.

2. Data Training:
a) Train-Test Split: The dataset was split into training and testing sets using the train_test_split function from sklearn.model_selection.                             This ensures that the model's performance can be evaluated on unseen data.
b) Model Training: Several regression algorithms were trained using the training data:
Linear Regression
Lasso Regression
Ridge Regression
Bagging Regression
Random Forest
LightGBM
c) Model Fitting: Each regression model was fitted to the training data using the fit method provided by the respective model classes. This involves learning the patterns and relationships between the input features and the target variable to predicting the number of rings in the abalone dataset which reflects the age.

3. Model Evaluation:
a) Predictions: After training the models, predictions were made on the test data using the predict method.
b) Evaluation Metrics: The performance of each model was evaluated using the root mean squared error (RMSE), a commonly used metric for regression tasks. The RMSE measures the average deviation of the predicted values from the actual values.
c) Comparison: The RMSE scores of all models were printed to compare their performance. This helps in selecting the best-performing model for the given dataset and task.
## 5.0 Dependencies

Install underlisted dependencies on your system:
1. pandas
2. numpy
3. scikit-learn (including LinearRegression, Lasso, Ridge, BaggingRegressor, RandomForestRegressor)
lightgbm
Install the dependencies using pip command:
pip install pandas numpy scikit-learn lightgbm



## 6.0 Installation
To install and run the Abalone Regression Predictiion program, follow these steps:
1. Clone the repository:
git clone https://github.com/hazzanolly1/Abalone Regression Prediction using Hassan Daud's Code.git
2. Ensure that you have an active internet connection to download the dataset from the UCI Machine Learning Repository.
3. Navigate to the Abalone Regression Prediction directory:
cd Abalone Regression Prediction using Hassan Daud's Code
4. Ensure you have Python 3.x installed. If not, download and install it from the official Python website.
5. Run the Python script:
Execute the Python script Abalone Regression Prediction using Hassan Daud's Code.py  containing the code for loading, preprocessing, training, and evaluating the models on the Abalone dataset.
6. Adjust the instructions as needed based on your actual implementation and requirements.
Run the Program:
7. View Results:
The program will print the Root Mean Squared Error (RMSE) for each regression model, indicating the accuracy of the predictions.
Execute the Python script containing the provided code. 


    
## 7.0 References
1. Abalone Dataset: UCI Machine Learning Repository "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data" 
2. Python Libraries: pandas, NumPy, scikit-learn, LightGBM
## 7.0 Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request with any improvements or new features.
Please adhere to this project's `code of conduct`.


## 8.0 License
This project is licensed under the MIT License.
