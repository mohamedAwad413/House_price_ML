import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Page Configuration
st.set_page_config(
    page_title="Universal Real Estate Price Predictor",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 Universal Real Estate Price Predictor & Visualizer")
st.write("This application dynamically adapts to any uploaded CSV dataset containing a 'Price' column, trains a **Random Forest Regressor**, and generates interactive tools.")

# 2. File Uploader
st.sidebar.header("📁 Dataset Management")
uploaded_file = st.sidebar.file_uploader("Upload any CSV dataset", type=["csv"])

@st.cache_data
def load_default_data():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'Area_SqFt': np.random.randint(800, 4000, size=n),
        'Bedrooms': np.random.randint(1, 5, size=n),
        'Bathrooms': np.random.randint(1, 4, size=n),
        'Price': np.random.randint(100000, 600000, size=n)
    })

# Load data safely
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        df = load_default_data()
else:
    df = load_default_data()
    st.sidebar.info("ℹ️ Using default dataset. Upload one of your custom CSV files.")

# 3. Check if 'Price' column exists
if 'Price' not in df.columns:
    st.error("⚠️ The uploaded CSV file must contain a column named 'Price' to train the model.")
else:
    # Exploratory Data Analysis & Visualizations
    with st.expander("📊 Dataset Overview & Visualizations"):
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        st.subheader("Dynamic Feature Scatter Chart")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        plotable_cols = [col for col in numeric_cols if col != 'Price']
        
        if plotable_cols:
            feature_to_plot = st.selectbox("Select Feature to plot against Price:", plotable_cols)
            st.scatter_chart(df, x=feature_to_plot, y='Price')
        else:
            st.warning("No numeric feature columns available for plotting.")

    # 4. Model Training
    X = df.drop(columns=['Price'])
    X_numeric = X.select_dtypes(include=[np.number])

    if X_numeric.empty:
        st.error("⚠️ The dataset must contain at least one numeric feature column besides 'Price'.")
    else:
        y = df['Price']

        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        with st.expander("📈 Academic Evaluation Metrics"):
            st.markdown(f"### **R² Score (Model Accuracy): {r2:.2f}%**")
            st.write(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f}")
            st.write("Evaluates regression metrics dynamically based on the uploaded features.")

        st.divider()
        st.subheader("📋 Predict Price with Dynamic Inputs:")

        user_inputs = {}
        cols = st.columns(2)
        
        # Dynamically generate widgets for ALL numeric columns in the dataset
        for idx, feature in enumerate(X_numeric.columns):
            col = cols[idx % 2]
            min_val = float(X_numeric[feature].min())
            max_val = float(X_numeric[feature].max())
            mean_val = float(X_numeric[feature].mean())
            
            if X_numeric[feature].nunique() <= 12:
                user_inputs[feature] = col.selectbox(f"{feature}:", sorted(X_numeric[feature].unique()))
            else:
                user_inputs[feature] = col.slider(f"{feature}:", min_value=min_val, max_value=max_val, value=mean_val)

        if st.button("Predict Property Price 🔍", type="primary"):
            input_df = pd.DataFrame([user_inputs])
            predicted_price = model.predict(input_df)[0]

            st.divider()
            st.subheader("Prediction Result:")
            st.success(f"🏡 **Estimated Property Price:** ${predicted_price:,.2f}")