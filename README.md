# Habit Tracker

A machine learning project that predicts HR employee attrition using a Random Forest classifier.

## Overview

This project analyzes employee data from a HR dataset and builds a predictive model to identify which employees are likely to leave the company.

## Features

- **Data Processing**: Loads and preprocesses HR employee data from CSV
- **Feature Engineering**: Encodes categorical variables (Attrition, Department, Job Role, etc.)
- **Machine Learning**: Uses scikit-learn's Random Forest Classifier for predictions
- **Data Normalization**: Implements MinMaxScaler for feature scaling

## Requirements

- pandas
- numpy
- scikit-learn

## Installation

```bash
pip install pandas numpy scikit-learn
```

## Usage

Run the prediction model:

```bash
python monitor
```

## How It Works

1. Loads HR employee attrition data from CSV
2. Handles missing values with dropna()
3. Encodes categorical variables using one-hot encoding
4. Splits data into training (80%) and testing (20%) sets
5. Trains a Random Forest Classifier
6. Makes predictions on test data
7. Outputs predictions

## Dataset

The project uses the WA Fn-UseC HR Employee Attrition dataset located at:
`companywork/WA_Fn-UseC_-HR-Employee-Attrition.csv`

## Model

- **Algorithm**: Random Forest Classifier
- **Test Size**: 20% of data
- **Random State**: 42 (for reproducibility)