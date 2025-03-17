# **Gradient Boosting Regressor**

Importing required libraries
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

"""Load dataset"""

# Load the dataset
data = pd.read_csv('crop_yield.csv')
data.head()

"""Encoding categorical data"""

# Convert categorical variables into numerical format using one-hot encoding
data_encoded = pd.get_dummies(data)

# Split the data into features (X) and target variable (y)
X = data_encoded.drop(columns=['Yield'])  # Features
y = data_encoded['Yield']  # Target variable

"""Test-Train dataset split"""

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Creating the model"""

# Create and train the GradientBoostingRegressor model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

"""Predictions"""

# Predict the yield on the test set
y_pred = model.predict(X_test)

"""Model Evaluation"""

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Root Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

"""Calculating Accuracy (In Precentage)"""

# Calculate the percentage of predictions within a tolerance
tolerance = 10  # Define a tolerance threshold
correct_predictions = sum(abs(y_test - y_pred) < tolerance)
total_predictions = len(y_test)
accuracy = (correct_predictions / total_predictions) * 100
print("Accuracy (within {}): {:.2f}%".format(tolerance, accuracy))

"""Visualization"""

import matplotlib.pyplot as plt

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Crop Yield (Gradien Boosting Regressor)')
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.grid(True)
plt.show()

"""Correlation Heatmap"""

# Drop the target variable and one-hot encoded columns
data_factors = data.drop(columns=['Yield']).select_dtypes(include=['float64', 'int64'])

# Calculate correlation matrix
correlation_matrix = data_factors.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5, linecolor='white')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.xlabel('Factors')
plt.ylabel('Factors')
plt.title('Correlation Heatmap of Factors')
plt.show()

"""# **Decision Tree Regressor**

Importing required libraries
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

"""Importing the dataset"""

# Load the dataset
data = pd.read_csv('crop_yield.csv')

"""Encoding categorical data"""

# Convert categorical variables into numerical format using one-hot encoding
data_encoded = pd.get_dummies(data)

# Split the data into features (X) and target variable (y)
X = data_encoded.drop(columns=['Yield'])  # Features
y = data_encoded['Yield']  # Target variable

"""Split dataset into testing and training"""

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Model creation"""

# Create and train the decision tree model
model = DecisionTreeRegressor(max_depth=10, random_state=42)
model.fit(X_train, y_train)

"""Predictions"""

# Predict the yield on the test set
y_pred = model.predict(X_test)

"""Model Evaluation"""

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

"""Decision Tree"""

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
plt.show()

"""Visualization"""

import matplotlib.pyplot as plt

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Crop Yield (Decision Tree Regressor)')
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.grid(True)
plt.show()

"""# **Support Vector Regressor**

Import all the required libraries
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

"""Importing the dataset"""

# Load the dataset
df = pd.read_csv("crop_yield.csv")

"""Feature Selection"""

# Split features and target variable
X = df[['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
y = df['Yield']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical features and numerical features
categorical_features = ['Crop', 'Season', 'State']
numerical_features = ['Crop_Year', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

"""Data Scaling"""

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Append estimator to preprocessing pipeline
svr_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', SVR(kernel='rbf'))])

"""Model Training"""

# Train SVR model
svr_model.fit(X_train, y_train)

"""Predictions"""

# Predictions
y_pred_train = svr_model.predict(X_train)
y_pred_test = svr_model.predict(X_test)

"""Model Evaluation"""

# Evaluate the model
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
test_rmae = mean_absolute_error(y_test, y_pred_test)

print(f"Test Mean Squared Error: {test_rmse:.2f}")
print(f"Test Mean Absolute Error: {test_rmae:.2f}")
