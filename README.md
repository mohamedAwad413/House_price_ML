# 🏡 Universal Real Estate Price Predictor

A simple Streamlit app that trains a **Random Forest Regressor** on any CSV dataset containing a `Price` column, then lets you explore the data and predict property prices interactively.

## Features

- 📁 Upload any CSV dataset (a sample dataset is used if nothing is uploaded)
- 📊 Data preview and an interactive scatter chart (feature vs. Price)
- 🤖 Automatic model training with an 80/20 train/test split
- 📈 Evaluation metrics: R² score and RMSE
- 🔍 Dynamic input widgets generated from your dataset's numeric columns
- 💰 Instant price prediction

## Requirements

- Python 3.9+
- streamlit
- pandas
- numpy
- scikit-learn

## Installation

```bash
pip install streamlit pandas numpy scikit-learn
```

## Usage

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Dataset Format

Your CSV file must:

1. Contain a column named **`Price`** (the target to predict).
2. Contain at least one **numeric** feature column besides `Price`.

Example:

| Area_SqFt | Bedrooms | Bathrooms | Price  |
|-----------|----------|-----------|--------|
| 1500      | 3        | 2         | 250000 |
| 2200      | 4        | 3         | 410000 |

> Non-numeric columns are ignored during training.

## How It Works

1. Load the uploaded CSV (or the default sample data).
2. Split the data into training (80%) and testing (20%) sets.
3. Train a `RandomForestRegressor` with 100 trees.
4. Show R² and RMSE on the test set.
5. Generate input widgets (sliders or dropdowns) for each numeric feature.
6. Predict the price from the values you choose.

## Project Structure

```
.
├── app.py
└── README.md
```

## License

Free to use for learning and personal projects.
