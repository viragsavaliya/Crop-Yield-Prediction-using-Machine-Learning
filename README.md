# Crop Yield Prediction

## Overview

This project aims to predict crop yield using various machine learning models. It includes implementations of Gradient Boosting Regressor, Decision Tree Regressor, and Support Vector Regressor. The goal is to provide accurate predictions of crop yield based on factors such as crop type, season, rainfall, and other relevant features.

## Table of Contents

- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Models](#models)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Contributing](#contributing)

## Dataset

The dataset used in this project is `crop_yield.csv`, which contains the following columns:
- **Crop**: Type of crop
- **Crop_Year**: Year of the crop
- **Season**: Season of the crop
- **State**: State where the crop is grown
- **Area**: Area of the crop (in hectares)
- **Production**: Production of the crop (in tons)
- **Annual_Rainfall**: Annual rainfall (in mm)
- **Fertilizer**: Amount of fertilizer used (in kg)
- **Pesticide**: Amount of pesticide used (in kg)
- **Yield**: Yield of the crop (target variable)

## Dependencies

To run this project, you need the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install these dependencies using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Models

### Gradient Boosting Regressor

This model uses the Gradient Boosting algorithm to predict crop yield. It provides robust performance and handles non-linear relationships well.

### Decision Tree Regressor

This model uses a Decision Tree algorithm to predict crop yield. It is simple to understand and interpret, making it a good baseline model.

### Support Vector Regressor

This model uses the Support Vector Machine algorithm with a radial basis function (RBF) kernel to predict crop yield. It is effective in handling high-dimensional spaces and non-linear relationships.

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/viragsavaliya/crop_yield_prediction.git
   cd crop_yield_prediction
   ```

2. **Prepare the dataset:**
   Ensure you have the `crop_yield.csv` file in the project directory.

3. **Run the script:**
   ```bash
   python crop_yield_prediction.py
   ```

4. **Evaluate the models:**
   The script will train the models and evaluate their performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## Evaluation

The models are evaluated using the following metrics:
- **Mean Squared Error (MSE):** Measures the average squared difference between predicted and actual values.
- **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual values.
- **Accuracy (within tolerance):** Percentage of predictions within a specified tolerance range.

## Visualization

The project includes visualizations to help understand the model performance:
- **Actual vs Predicted Yield:** Scatter plot showing the actual vs predicted crop yield.
- **Correlation Heatmap:** Heatmap showing the correlation between different factors.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.
