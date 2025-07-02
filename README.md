# F1 Pitstop Champion Prediction

This repository contains a Python pipeline to predict Formula 1 season champions using pitstop and race data from 2018 to 2024. It covers data preprocessing, feature engineering, exploratory visualization, model training, and evaluation.

## Project Overview

Predict which driver will become the Formula 1 World Champion next season by leveraging pitstop performance, weather, and race metrics. The pipeline trains four classifiers—Logistic Regression, Random Forest, Naive Bayes, and KNN—and compares their performance.

## Pipeline Steps

### 1. Load & Encode Drivers

- Read the CSV into a pandas DataFrame.
- Convert driver names into numeric `DriverCode` using `pd.factorize`.

### 2. Compute Season Points & Champions

- Map finishing positions to FIA points (1st→25, 2nd→18, …).
- Sum points per driver per season.
- Flag the driver with the highest points in each season as champion (`IsChampion`).

### 3. Clean & Drop Columns

- Remove unneeded text columns (e.g., race name, date, location).
- Retain only numeric and categorical features relevant for modeling.

### 4. Handle Missing Values

- **Numeric columns**: fill missing values with column medians.
- **Categorical columns**: fill missing values with the mode (most frequent value).

### 5. Aggregate Features

- Group by `Season` and `DriverCode` to get one row per driver-season.
- Compute:
  - **Numeric**: mean or sum for continuous metrics (e.g., average pit stops, lap variation, total laps).
  - **Mode**: most common `Constructor` and `Tire Compound`.
  - **Label**: max of `IsChampion` (1 if the driver won that season).

### 6. Exploratory Visualizations

- **Correlation Matrix**: visualizes inter-feature correlations.
- **Feature Importances**: top 10 predictive features from a small Random Forest.

### 7. Feature Selection & Importances

- Use `SelectKBest` (ANOVA F-test) to score all features.
- Plot the top features.

### 8. Prepare the ML Dataset

- Create the `ChampionNext` label by shifting `IsChampion` to the next season.
- Split the data into:
  - **Training**: seasons ≤ 2022.
  - **Test**: season 2023.
- Remove identifiers (`Season`, `DriverCode`) from the feature set.

### 9. Model Training & Evaluation

- Balance classes in the training set with `RandomOverSampler`.
- Train and evaluate four models:
  - **Logistic Regression**
  - **Random Forest** (max_depth=5)
  - **Gaussian Naive Bayes**
  - **K-Nearest Neighbors** (n=5)
- Evaluate on 2023 test data:
  - Accuracy and weighted F1 score.
  - Classification report.
  - Confusion matrix.

| Model               | Accuracy | F1 Score |
|---------------------|----------|----------|
| Logistic Regression | 1.000    | 1.000    |
| Random Forest       | 1.000    | 1.000    |
| Naive Bayes         | 1.000    | 1.000    |
| KNN                 | 0.429    | 0.551    |

### 10. Final Model & 2025 Prediction

- Retrain the best model on all data up to 2023.
- Predict champion probabilities for each driver in the 2024 season.
- Output the top-5 drivers by probability.

## License

MIT © 2025
