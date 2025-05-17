import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
from PIL import Image
import requests
from io import BytesIO
import os
from sklearn.impute import SimpleImputer

# Set page configuration
st.set_page_config(
    page_title="Multi-Themed Financial ML App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define theme colors
zombie_colors = {
    "primary": "#2D0000",
    "secondary": "#8B0000",
    "background": "#000000",
    "text": "#FF0000",
    "accent": "#4A0000"
}

futuristic_colors = {
    "primary": "#00FFFF",
    "secondary": "#FF00FF",
    "background": "#000033",
    "text": "#FFFFFF",
    "accent": "#00FF00"
}

got_colors = {
    "primary": "#990000",
    "secondary": "#000066",
    "background": "#1A1A1A",
    "text": "#D4AF37",
    "accent": "#800000"
}

# Helper functions for theming
def set_zombie_theme():
    """Apply zombie theme styling"""
    # CSS for zombie theme
    zombie_css = f"""
    <style>
        .zombie-section {{
            background-color: {zombie_colors["background"]};
            color: {zombie_colors["text"]};
            padding: 20px;
            border-radius: 5px;
            border: 2px solid {zombie_colors["secondary"]};
            font-family: 'Creepster', cursive;
        }}
        .zombie-header {{
            color: {zombie_colors["text"]};
            text-shadow: 2px 2px 4px {zombie_colors["accent"]};
            font-family: 'Creepster', cursive;
        }}
        .zombie-subheader {{
            color: {zombie_colors["secondary"]};
            font-family: 'Creepster', cursive;
        }}
        @import url('https://fonts.googleapis.com/css2?family=Creepster&display=swap');
    </style>
    """
    st.markdown(zombie_css, unsafe_allow_html=True)
    
def set_futuristic_theme():
    """Apply futuristic theme styling"""
    # CSS for futuristic theme
    futuristic_css = f"""
    <style>
        .futuristic-section {{
            background-color: {futuristic_colors["background"]};
            color: {futuristic_colors["text"]};
            padding: 20px;
            border-radius: 5px;
            border: 2px solid {futuristic_colors["primary"]};
            box-shadow: 0 0 10px {futuristic_colors["primary"]};
            font-family: 'Orbitron', sans-serif;
        }}
        .futuristic-header {{
            color: {futuristic_colors["primary"]};
            text-shadow: 0 0 5px {futuristic_colors["primary"]};
            font-family: 'Orbitron', sans-serif;
        }}
        .futuristic-subheader {{
            color: {futuristic_colors["secondary"]};
            font-family: 'Orbitron', sans-serif;
        }}
        @import url('https://fonts.googleapis.com/css2?family=Orbitron&display=swap');
    </style>
    """
    st.markdown(futuristic_css, unsafe_allow_html=True)

def set_got_theme():
    """Apply Game of Thrones theme styling"""
    # CSS for Game of Thrones theme
    got_css = f"""
    <style>
        .got-section {{
            background-color: {got_colors["background"]};
            color: {got_colors["text"]};
            padding: 20px;
            border-radius: 5px;
            border: 2px solid {got_colors["primary"]};
            font-family: 'MedievalSharp', cursive;
        }}
        .got-header {{
            color: {got_colors["text"]};
            text-shadow: 2px 2px 4px {got_colors["secondary"]};
            font-family: 'MedievalSharp', cursive;
        }}
        .got-subheader {{
            color: {got_colors["primary"]};
            font-family: 'MedievalSharp', cursive;
        }}
        @import url('https://fonts.googleapis.com/css2?family=MedievalSharp&display=swap');
    </style>
    """
    st.markdown(got_css, unsafe_allow_html=True)

# Function to display themed header
def themed_header(title, theme):
    if theme == "zombie":
        st.markdown(f'<h1 class="zombie-header">{title}</h1>', unsafe_allow_html=True)
    elif theme == "futuristic":
        st.markdown(f'<h1 class="futuristic-header">{title}</h1>', unsafe_allow_html=True)
    elif theme == "got":
        st.markdown(f'<h1 class="got-header">{title}</h1>', unsafe_allow_html=True)

# Function to display themed subheader
def themed_subheader(title, theme):
    if theme == "zombie":
        st.markdown(f'<h2 class="zombie-subheader">{title}</h2>', unsafe_allow_html=True)
    elif theme == "futuristic":
        st.markdown(f'<h2 class="futuristic-subheader">{title}</h2>', unsafe_allow_html=True)
    elif theme == "got":
        st.markdown(f'<h2 class="got-subheader">{title}</h2>', unsafe_allow_html=True)

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Function to load local image
def load_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Welcome Page CSS
welcome_css = """
<style>
    /* Base styles and animations */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    .welcome-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
        text-align: center;
    }
    
    .welcome-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem !important;
        font-weight: 700;
        background: linear-gradient(120deg, #00ff00, #00ffff, #d4af37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeIn 1.5s ease-out;
    }
    
    .welcome-subtitle {
        font-family: 'Poppins', sans-serif;
        color: #cccccc;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        animation: fadeIn 2s ease-out;
    }
    
    .theme-card {
        background: rgba(0, 0, 0, 0.6);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideUp 1s ease-out;
        border: 1px solid rgba(255, 255, 255, 0.1);
        height: 100%;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .theme-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
    
    .theme-card:hover .theme-icon {
        transform: scale(1.2) rotate(10deg);
    }
    
    .theme-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
        display: inline-block;
    }
    
    .theme-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        font-size: 0.8rem;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    
    .theme-title {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    
    .theme-desc {
        color: #cccccc;
        margin-bottom: 1.5rem;
    }
    
    .btn-explore {
        background: linear-gradient(120deg, #00ff00, #00ffff, #d4af37);
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        color: #000000;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .btn-explore:hover {
        opacity: 0.9;
        transform: scale(1.05);
    }
    
    .btn-explore::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: rgba(255, 255, 255, 0.1);
        transform: rotate(45deg);
        transition: transform 0.5s ease;
    }
    
    .btn-explore:hover::after {
        transform: rotate(45deg) translate(50%, 50%);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .floating-emoji {
        font-size: 2rem;
        animation: float 3s ease-in-out infinite;
        display: inline-block;
        margin: 0 0.5rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .welcome-title {
            font-size: 2.5rem !important;
        }
        .welcome-subtitle {
            font-size: 1rem;
        }
    }
</style>
"""

# Main app
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a section:",
        ["Welcome", "Zombie Theme: Linear Regression", "Futuristic Theme: Logistic Regression", "GoT Theme: K-Means Clustering"]
    )
    
    # Welcome Page
    if app_mode == "Welcome":
        st.markdown(welcome_css, unsafe_allow_html=True)
        st.markdown('<div class="welcome-container">', unsafe_allow_html=True)
        
        # Add floating emojis around the title
        st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <span class="floating-emoji">🚀</span>
                <span class="floating-emoji">💹</span>
                <span class="floating-emoji">📈</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h1 class="welcome-title">Financial ML Dashboard</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="welcome-subtitle">Explore financial data through immersive themed experiences powered by machine learning</p>', 
            unsafe_allow_html=True
        )
        
        # Create theme cards in a grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="theme-card">
                    <div class="theme-icon">🧟</div>
                    <div class="theme-badge">Linear Regression</div>
                    <h3 class="theme-title">Zombie Apocalypse</h3>
                    <p class="theme-desc">Survive financial uncertainty with predictive analytics that help you prepare for market downturns.</p>
                    <button class="btn-explore" onclick="document.querySelector('[data-testid=stSidebar] [data-testid=stSidebarNav]').click()">Enter the Apocalypse</button>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="theme-card">
                    <div class="theme-icon">🤖</div>
                    <div class="theme-badge">Classification Models</div>
                    <h3 class="theme-title">Cyber Future</h3>
                    <p class="theme-desc">Harness artificial intelligence to predict market movements and classify investment risks.</p>
                    <button class="btn-explore" onclick="document.querySelector('[data-testid=stSidebar] [data-testid=stSidebarNav]').click()">Enter the Future</button>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="theme-card">
                    <div class="theme-icon">👑</div>
                    <div class="theme-badge">K-Means Clustering</div>
                    <h3 class="theme-title">Game of Thrones</h3>
                    <p class="theme-desc">Uncover hidden patterns in the market like the secrets of the Seven Kingdoms.</p>
                    <button class="btn-explore" onclick="document.querySelector('[data-testid=stSidebar] [data-testid=stSidebarNav]').click()">Enter the Realm</button>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add interactive theme selection with emojis
        st.markdown("""
            <div style="text-align: center; margin-top: 2rem;">
                <h3 style="color: #ffffff; margin-bottom: 1rem;">Choose Your Experience</h3>
            </div>
        """, unsafe_allow_html=True)
        
        theme_col1, theme_col2, theme_col3 = st.columns(3)
        
        with theme_col1:
            if st.button("🧟 Zombie Apocalypse", use_container_width=True):
                st.session_state.current_theme = "Zombie Theme: Linear Regression"
                st.rerun()
                
        with theme_col2:
            if st.button("🤖 Cyber Future", use_container_width=True):
                st.session_state.current_theme = "Futuristic Theme: Logistic Regression"
                st.rerun()
                
        with theme_col3:
            if st.button("👑 Game of Thrones", use_container_width=True):
                st.session_state.current_theme = "GoT Theme: K-Means Clustering"
                st.rerun()
    
    # Zombie Theme Section - Linear Regression
    elif app_mode == "Zombie Theme: Linear Regression":
        from datetime import datetime, timedelta
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        import plotly.graph_objects as go

        set_zombie_theme()

        st.markdown('<div class="zombie-section">', unsafe_allow_html=True)
        
        # Center the header
        st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 class="zombie-header">Zombie Market Prediction</h1>
            </div>
        """, unsafe_allow_html=True)

        # Center the image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            try:
                st.image("images/zombie.jpg", width=300)
            except Exception as e:
                st.warning("Please add zombie.jpg to the images folder")

        themed_subheader("Stock Price Prediction with Linear Regression", "zombie")
        st.markdown("<p style='color: #FF0000;'>Predict the apocalyptic future of stock prices using the dark arts of linear regression...</p>", unsafe_allow_html=True)

        st.markdown("<h3 style='color: #8B0000;'>Summon Your Stock Data</h3>", unsafe_allow_html=True)

        data_source = st.radio("Choose your data source of doom:", ["Upload CSV", "Fetch from Yahoo Finance"])

        df = None

        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your cursed CSV file", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success("Data has been summoned successfully!")
                st.dataframe(df.head())
        else:
            ticker = st.text_input("Enter the stock ticker symbol (e.g., AAPL):", "AAPL")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
            with col2:
                end_date = st.date_input("End Date", datetime.now())

            if st.button("Summon Stock Data"):
                with st.spinner("Summoning data from the netherworld..."):
                    df = fetch_stock_data(ticker, start_date, end_date)
                    if df is not None and not df.empty:
                        st.success("Stock data has been conjured!")
                        df.reset_index(inplace=True)
                        st.dataframe(df.head())
                    else:
                        st.error("Failed to summon data. Check the ticker or date range.")

        if df is not None:
            st.markdown("<h3 style='color: #8B0000;'>Select Features for the Ritual</h3>", unsafe_allow_html=True)

            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days

            # Get numeric columns more explicitly
            numeric_columns = []
            for col in df.columns:
                try:
                    # Try to convert to numeric, ignoring errors
                    pd.to_numeric(df[col], errors='coerce')
                    if col != 'Date':  # Exclude Date from numeric columns
                        numeric_columns.append(col)
                except:
                    continue

            if numeric_columns:
                st.write("Available numeric columns:", numeric_columns)
                
                # Ensure we have a valid target column
                if 'Price' in numeric_columns:
                    target_col = 'Price'
                elif 'Close' in numeric_columns:
                    target_col = 'Close'
                else:
                    target_col = st.selectbox("Select the target column to predict:", numeric_columns)

                # Get available features (excluding target column)
                available_features = [col for col in numeric_columns if col != target_col]
                
                if not available_features:
                    st.error("No valid features available for prediction!")
                else:
                    # Set default features if available
                    default_features = [col for col in ['Open', 'High', 'Low', 'Volume'] if col in available_features]
                    if not default_features:
                        default_features = [available_features[0]]  # Use first available feature as default
                    
                    st.write("Available features for selection:", available_features)
                    st.write("Default features:", default_features)
                    
                    # Initialize feature_cols with default features
                    if 'feature_cols' not in st.session_state:
                        st.session_state.feature_cols = default_features
                    
                    feature_cols = st.multiselect(
                        "Select feature columns for your dark prediction:",
                        options=available_features,
                        default=st.session_state.feature_cols,
                        key="linear_regression_features"
                    )
                    
                    # Update session state with selected features
                    if feature_cols:
                        st.session_state.feature_cols = feature_cols
                    else:
                        # If no features selected, use default features
                        feature_cols = default_features
                        st.session_state.feature_cols = default_features

                    test_size = st.slider("Test size (percentage of data for validation)", 10, 50, 20)

                    if st.button("Perform the Linear Regression Ritual"):
                        try:
                            with st.spinner("The dark calculation is in progress..."):
                                # Debug information
                                st.write("Selected features:", feature_cols)
                                st.write("Target column:", target_col)
                                
                                # Convert columns to numeric type and handle special formats
                                for col in feature_cols + [target_col]:
                                    if col in df.columns:
                                        # Handle percentage values
                                        if '%' in col:
                                            df[col] = df[col].str.rstrip('%').astype('float') / 100.0
                                        # Handle volume with K format
                                        elif 'Vol' in col:
                                            df[col] = df[col].astype(str).apply(lambda x: float(x.replace('K', '').replace(',', '')) * 1000 if 'K' in x else float(x.replace(',', '')))
                                        else:
                                            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
                                
                                X = df[feature_cols]
                                y = df[target_col]

                                # Debug information
                                st.write("X shape:", X.shape)
                                st.write("y shape:", y.shape)
                                
                                if X.shape[1] == 0:
                                    st.error("No features were selected for the model!")
                                    return

                                # Create and fit imputer
                                imputer = SimpleImputer(strategy='mean')
                                X_imputed = imputer.fit_transform(X)
                                y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()
                                
                                # Debug information
                                st.write("X_imputed shape:", X_imputed.shape)
                                st.write("y_imputed shape:", y_imputed.shape)
                                
                                # Split data
                                X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=test_size / 100, random_state=42)

                                model = LinearRegression()
                                model.fit(X_train, y_train)

                                y_pred_train = model.predict(X_train)
                                y_pred_test = model.predict(X_test)

                                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"<h4 style='color: #8B0000;'>Training RMSE: {train_rmse:.4f}</h4>", unsafe_allow_html=True)
                                with col2:
                                    st.markdown(f"<h4 style='color: #FF0000;'>Testing RMSE: {test_rmse:.4f}</h4>", unsafe_allow_html=True)

                                st.markdown("<h4 style='color: #8B0000;'>Feature Importance (Coefficients)</h4>", unsafe_allow_html=True)
                                coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': model.coef_})

                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.bar(coef_df['Feature'], coef_df['Coefficient'], color='darkred')
                                ax.set_xlabel('Features', color='red')
                                ax.set_ylabel('Coefficient Value', color='red')
                                ax.set_title('Feature Importance', color='red')
                                ax.tick_params(colors='red')
                                for spine in ax.spines.values():
                                    spine.set_color('red')
                                fig.patch.set_facecolor('black')
                                ax.set_facecolor('black')

                                st.pyplot(fig)

                                st.markdown("<h4 style='color: #8B0000;'>Actual vs. Predicted Values</h4>", unsafe_allow_html=True)

                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='Actual', line=dict(color='red')))
                                fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_pred_test, mode='lines', name='Predicted', line=dict(color='darkred', dash='dash')))
                                fig.update_layout(
                                    title='Actual vs Predicted Values',
                                    xaxis_title='Data Point Index',
                                    yaxis_title=f'{target_col} Value',
                                    template='plotly_dark',
                                    plot_bgcolor='black',
                                    paper_bgcolor='black',
                                    font=dict(color='red')
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                st.markdown("<h4 style='color: #8B0000;'>Predict the Doomed Future</h4>", unsafe_allow_html=True)

                                input_values = {}
                                for feature in feature_cols:
                                    min_val = float(df[feature].min())
                                    max_val = float(df[feature].max())
                                    mean_val = float(df[feature].mean())

                                    input_values[feature] = st.slider(
                                        f"Value for {feature}",
                                        min_val, max_val, mean_val,
                                        step=(max_val - min_val) / 100 or 1.0  # Avoid step=0
                                    )

                                if st.button("Predict the Future Horror"):
                                    input_df = pd.DataFrame([input_values])
                                    prediction = model.predict(input_df)[0]

                                    st.markdown(
                                        f"<div style='background-color: #300; padding: 20px; border-radius: 5px; text-align: center;'>"
                                        f"<h3 style='color: #FF0000;'>The prophecy reveals that {target_col} will be:</h3>"
                                        f"<h2 style='color: #FF0000; font-size: 3em;'>${prediction:.2f}</h2>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                        except Exception as e:
                            st.error(f"Error during calculation: {str(e)}")
                            st.write("Debug information:")
                            st.write("Feature columns:", feature_cols)
                            st.write("DataFrame columns:", df.columns.tolist())
            else:
                st.error("No numeric columns found in the data. The ritual cannot continue!")

        st.markdown('</div>', unsafe_allow_html=True)
    
    # Futuristic Theme Section - Logistic Regression
    elif app_mode == "Futuristic Theme: Logistic Regression":
        set_futuristic_theme()
        
        # Import required libraries
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        import plotly.graph_objects as go
        import plotly.express as px
        
        st.markdown('<div class="futuristic-section">', unsafe_allow_html=True)
        
        # Center the header
        st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 class="futuristic-header">Quantum Market Analysis</h1>
            </div>
        """, unsafe_allow_html=True)
        
        # Center the image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            try:
                st.image("images/futuristic.jpg", width=300)
            except:
                st.warning("Please add futuristic.jpg to the images folder")
        
        themed_subheader("Market Trend Classification with Logistic Regression", "futuristic")
        st.markdown("<p style='color: #00FFFF;'>Use neural networks and quantum algorithms to classify market movements...</p>", unsafe_allow_html=True)
        
        # Data input
        st.markdown("<h3 style='color: #00FFFF;'>Initialize Data Sequence</h3>", unsafe_allow_html=True)
        
        # Input choices
        data_source = st.radio("Select data acquisition method:", ["Upload CSV", "Fetch from Yahoo Finance"], key="futuristic_data_source")
        
        df = None
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload financial data file", type=["csv"], key="futuristic_uploader")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Show the first few rows of raw data
                    st.write("Raw data preview:")
                    st.dataframe(df.head())
                    
                    # Convert column names to lowercase for case-insensitive matching
                    df.columns = df.columns.str.lower()
                    
                    # Check for required columns with case-insensitive matching
                    if 'price' in df.columns:
                        df = df.rename(columns={'price': 'close'})
                    
                    required_columns = ['date', 'close']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                        df = None
                    else:
                        # Convert numeric columns to float
                        numeric_columns = ['close', 'open', 'high', 'low', 'vol.']
                        for col in numeric_columns:
                            if col in df.columns:
                                # Convert K format and remove commas
                                df[col] = df[col].astype(str).apply(lambda x: float(x.replace('K', '').replace(',', '')) * 1000 if 'K' in x else float(x.replace(',', '')))
                        
                        # Convert date column to datetime
                        df['date'] = pd.to_datetime(df['date'])
                        
                        # Rename columns to standard format
                        df = df.rename(columns={
                            'date': 'Date',
                            'close': 'Close'
                        })
                        st.success("Data initialization complete!")
                        st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    df = None
        else:
            ticker = st.text_input("Enter stock symbol for analysis:", "TSLA", key="futuristic_ticker")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Starting temporal coordinate", datetime.now() - timedelta(days=365), key="futuristic_start")
            with col2:
                end_date = st.date_input("Ending temporal coordinate", datetime.now(), key="futuristic_end")
            
            if st.button("Initialize Data Download"):
                with st.spinner("Establishing quantum connection..."):
                    df = fetch_stock_data(ticker, start_date, end_date)
                    if df is not None and not df.empty:
                        st.success("Data successfully retrieved from the mainframe!")
                        df.reset_index(inplace=True)
                        st.dataframe(df.head())
                    else:
                        st.error("Failed to retrieve data. Please check the ticker symbol and try again.")
                        df = None
        
        if df is not None and not df.empty:
            # Create binary target variable
            st.markdown("<h3 style='color: #00FFFF;'>Define Classification Parameters</h3>", unsafe_allow_html=True)
            
            # Ensure Date and Close columns exist
            if 'Date' not in df.columns or 'Close' not in df.columns:
                st.error("Required columns 'Date' and 'Close' not found in the data.")
                st.info("Please ensure your data contains these columns for analysis.")
            else:
                # Create trend indicators
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values('Date')
                    df['PrevClose'] = df['Close'].shift(1)
                    df['Return'] = (df['Close'] - df['PrevClose']) / df['PrevClose']
                    
                    # Define threshold for classification
                    threshold = st.slider(
                        "Return threshold for positive classification (%):",
                        min_value=-5.0,
                        max_value=5.0,
                        value=0.0,
                        step=0.1,
                        key="futuristic_threshold"
                    )
                    
                    # Create target variable
                    df['Target'] = (df['Return'] > threshold/100).astype(int)
                    df = df.dropna()
                    
                    # Show distribution of classes
                    st.markdown("<h4 style='color: #00FFFF;'>Target Class Distribution</h4>", unsafe_allow_html=True)
                    class_counts = df['Target'].value_counts()
                    
                    fig = px.pie(values=class_counts.values, names=class_counts.index.map({0: 'Negative/Neutral', 1: 'Positive'}),
                                title='Target Class Distribution',
                                color_discrete_sequence=['#FF00FF', '#00FFFF'])
                    
                    fig.update_layout(
                        font=dict(color="#FFFFFF"),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature selection
                    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                    excluded_cols = ['Target', 'Return', 'PrevClose']
                    available_features = [col for col in numeric_columns if col not in excluded_cols]
                    
                    feature_cols = st.multiselect(
                        "Select features for the classification model:",
                        available_features,
                        default=[col for col in ['Open', 'High', 'Low', 'Volume'] if col in available_features][:3],
                        key="futuristic_features"
                    )
                    
                    # Add technical indicators option
                    if st.checkbox("Add technical indicators as features", key="futuristic_add_tech"):
                        with st.spinner("Calculating quantum technical indicators..."):
                            # Simple Moving Average
                            window_size = st.slider("SMA Window Size", 5, 50, 20, key="futuristic_sma")
                            df[f'SMA_{window_size}'] = df['Close'].rolling(window=window_size).mean()
                            
                            # Exponential Moving Average
                            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
                            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
                            
                            # MACD
                            df['MACD'] = df['EMA_12'] - df['EMA_26']
                            
                            # Relative Strength Index (simplified)
                            delta = df['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            df['RSI'] = 100 - (100 / (1 + rs))
                            
                            # Add new features to selection
                            tech_features = [f'SMA_{window_size}', 'EMA_12', 'EMA_26', 'MACD', 'RSI']
                            additional_features = st.multiselect(
                                "Select technical indicators to include:",
                                tech_features,
                                default=tech_features[:2],
                                key="futuristic_tech_select"
                            )
                            
                            feature_cols.extend(additional_features)
                            
                            # Show one technical indicator
                            if len(additional_features) > 0:
                                selected_indicator = st.selectbox(
                                    "Visualize technical indicator:",
                                    additional_features,
                                    key="futuristic_indicator_viz"
                                )
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=df['Date'],
                                    y=df[selected_indicator],
                                    mode='lines',
                                    name=selected_indicator,
                                    line=dict(color='#00FFFF', width=2)
                                ))
                                
                                fig.update_layout(
                                    title=f'{selected_indicator} Over Time',
                                    xaxis_title='Date',
                                    yaxis_title='Value',
                                    template='plotly_dark',
                                    plot_bgcolor='rgba(0,0,51,0.8)',
                                    paper_bgcolor='rgba(0,0,51,0.8)',
                                    font=dict(color='#00FFFF')
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Train-test split slider
                    test_size = st.slider("Test data allocation (%):", 10, 50, 20, key="futuristic_test_size")
                    
                    if st.button("Initialize Logistic Regression Algorithm"):
                        if len(feature_cols) > 0:
                            with st.spinner("Quantum processing in progress..."):
                                df = df.dropna()
                                
                                # Prepare data
                                X = df[feature_cols]
                                y = df['Target']
                                
                                # Standardize features
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                
                                # Split data
                                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size/100, random_state=42, stratify=y)
                                
                                # Train model
                                model = LogisticRegression(max_iter=1000, random_state=42)
                                model.fit(X_train, y_train)
                                
                                # Make predictions
                                y_pred_train = model.predict(X_train)
                                y_pred_test = model.predict(X_test)
                                y_pred_proba_test = model.predict_proba(X_test)[:, 1]
                                
                                # Calculate accuracy
                                train_accuracy = accuracy_score(y_train, y_pred_train)
                                test_accuracy = accuracy_score(y_test, y_pred_test)
                                
                                # Display results
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(
                                        f"<div style='background-color: rgba(0, 255, 255, 0.1); padding: 20px; border-radius: 10px; "
                                        f"border: 1px solid #00FFFF; text-align: center;'>"
                                        f"<h3 style='color: #00FFFF;'>Training Accuracy</h3>"
                                        f"<h2 style='color: #00FFFF; font-size: 2.5em;'>{train_accuracy:.2%}</h2>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                                
                                with col2:
                                    st.markdown(
                                        f"<div style='background-color: rgba(255, 0, 255, 0.1); padding: 20px; border-radius: 10px; "
                                        f"border: 1px solid #FF00FF; text-align: center;'>"
                                        f"<h3 style='color: #FF00FF;'>Testing Accuracy</h3>"
                                        f"<h2 style='color: #FF00FF; font-size: 2.5em;'>{test_accuracy:.2%}</h2>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                                
                                # Display coefficients
                                st.markdown("<h4 style='color: #00FFFF;'>Feature Importance Analysis</h4>", unsafe_allow_html=True)
                                coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': model.coef_[0]})
                                coef_df = coef_df.sort_values('Coefficient', ascending=False)
                                
                                fig = px.bar(
                                    coef_df,
                                    x='Feature',
                                    y='Coefficient',
                                    title='Feature Importance (Coefficient Magnitude)',
                                    color='Coefficient',
                                    color_continuous_scale=['#FF00FF', '#FFFFFF', '#00FFFF']
                                )
                                
                                fig.update_layout(
                                    xaxis_title='Feature',
                                    yaxis_title='Coefficient Value',
                                    template='plotly_dark',
                                    plot_bgcolor='rgba(0,0,51,0.8)',
                                    paper_bgcolor='rgba(0,0,51,0.8)',
                                    font=dict(color='#00FFFF')
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Probability distribution
                                st.markdown("<h4 style='color: #00FFFF;'>Prediction Probability Distribution</h4>", unsafe_allow_html=True)
                                
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(
                                    x=y_pred_proba_test,
                                    nbinsx=20,
                                    name='Prediction Probabilities',
                                    marker_color='#00FFFF'
                                ))
                                
                                fig.update_layout(
                                    title='Distribution of Prediction Probabilities',
                                    xaxis_title='Probability of Positive Class',
                                    yaxis_title='Count',
                                    template='plotly_dark',
                                    plot_bgcolor='rgba(0,0,51,0.8)',
                                    paper_bgcolor='rgba(0,0,51,0.8)',
                                    font=dict(color='#00FFFF')
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add prediction interface
                                st.markdown("<h4 style='color: #00FFFF;'>Quantum Prediction Interface</h4>", unsafe_allow_html=True)
                                
                                # Initialize session state for prediction if not exists
                                if 'show_prediction' not in st.session_state:
                                    st.session_state.show_prediction = False
                                
                                # Create input fields for each feature
                                input_values = {}
                                for feature in feature_cols:
                                    min_val = float(df[feature].min())
                                    max_val = float(df[feature].max())
                                    mean_val = float(df[feature].mean())
                                    
                                    input_values[feature] = st.slider(
                                        f"Value for {feature}",
                                        min_val, max_val, mean_val,
                                        step=(max_val-min_val)/100,
                                        key=f"futuristic_input_{feature}"
                                    )
                                
                                # Prediction button
                                if st.button("Generate Quantum Prediction", key="futuristic_predict"):
                                    st.session_state.show_prediction = True
                                
                                # Show prediction results if button was clicked
                                if st.session_state.show_prediction:
                                    try:
                                        # Create input dataframe for prediction
                                        input_df = pd.DataFrame([input_values])
                                        
                                        # Scale the input
                                        input_scaled = scaler.transform(input_df)
                                        
                                        # Make prediction
                                        prediction = model.predict(input_scaled)[0]
                                        probability = model.predict_proba(input_scaled)[0][1]
                                        
                                        # Display prediction with futuristic theme
                                        st.markdown(
                                            f"<div style='background-color: rgba(0, 255, 255, 0.1); padding: 20px; border-radius: 10px; "
                                            f"border: 1px solid #00FFFF; text-align: center;'>"
                                            f"<h3 style='color: #00FFFF;'>Quantum Analysis Result</h3>"
                                            f"<h2 style='color: #00FFFF; font-size: 2.5em;'>"
                                            f"{'POSITIVE' if prediction == 1 else 'NEGATIVE'} TREND</h2>"
                                            f"<p style='color: #00FFFF; font-size: 1.2em;'>"
                                            f"Confidence: {probability:.2%}</p>"
                                            f"</div>",
                                            unsafe_allow_html=True
                                        )
                                        
                                        # Add a button to reset the prediction
                                        if st.button("Make Another Prediction", key="reset_prediction"):
                                            st.session_state.show_prediction = False
                                            st.rerun()
                                            
                                    except Exception as e:
                                        st.error(f"Error generating prediction: {str(e)}")
                                        st.session_state.show_prediction = False
                        else:
                            st.error("You must select at least one feature for the quantum analysis!")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
        else:
            st.error("Please upload a dataset first!")

        st.markdown('</div>', unsafe_allow_html=True)        
        
# Game of Thrones Theme: K-Means Clustering
    elif app_mode == "GoT Theme: K-Means Clustering":
        set_got_theme()

        # Import required libraries
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        import plotly.express as px

        # Create a container with custom styling
        st.markdown("""
            <div class="got-section" style="
                background: linear-gradient(rgba(26,26,26,0.9), rgba(26,26,26,0.9)), 
                            url('https://images.unsplash.com/photo-1518709268805-4e9042af9f23?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80');
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 0 20px rgba(212, 175, 55, 0.5);
                border: 1px solid #D4AF37;
            ">
        """, unsafe_allow_html=True)

        # Enhanced header with animation
        st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="
                    color: #D4AF37;
                    font-size: 2.5em;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
                ">House Clustering Analysis</h1>
            </div>
        """, unsafe_allow_html=True)

        # Display GoT image with proper fallback
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            try:
                st.image("images/got.jpg", use_container_width=True, caption="Winter is Coming...")
            except FileNotFoundError:
                st.warning("Image not found: Please add 'got.jpg' to the 'images' folder!")

        # Subheader with styling
        st.markdown("""
            <div style="
                text-align: center;
                margin: 2rem 0;
                padding: 1rem;
                background: rgba(212, 175, 55, 0.1);
                border-radius: 10px;
            ">
                <h2 style="color: #D4AF37;">Stock Performance Clustering with K-Means</h2>
                <p style="color: #D4AF37; font-style: italic;">Discover the houses of the stock market through the ancient art of clustering...</p>
            </div>
        """, unsafe_allow_html=True)

        # Data input section
        st.markdown("""
            <div style="
                background: rgba(26,26,26,0.8);
                padding: 1.5rem;
                border-radius: 10px;
                border: 1px solid #D4AF37;
                margin: 1rem 0;
            ">
                <h3 style="color: #D4AF37; text-align: center;">Gather Your Market Intelligence</h3>
            </div>
        """, unsafe_allow_html=True)

        # Initialize df variable
        df = None

        # Input choices
        data_source = st.radio("Select data acquisition method:", ["Upload CSV", "Fetch from Yahoo Finance"], key="got_data_source")
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload financial data file", type=["csv"], key="got_uploader")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Convert column names to lowercase for case-insensitive matching
                    df.columns = df.columns.str.lower()
                    
                    # Check for required columns with case-insensitive matching
                    if 'price' in df.columns:
                        df = df.rename(columns={'price': 'close'})
                    
                    required_columns = ['date', 'close']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                        df = None
                    else:
                        # Convert numeric columns to float
                        numeric_columns = ['close', 'open', 'high', 'low', 'vol.']
                        for col in numeric_columns:
                            if col in df.columns:
                                # Convert K format and remove commas
                                df[col] = df[col].astype(str).apply(lambda x: float(x.replace('K', '').replace(',', '')) * 1000 if 'K' in x else float(x.replace(',', '')))
                        
                        # Convert date column to datetime
                        df['date'] = pd.to_datetime(df['date'])
                        
                        # Rename columns to standard format
                        df = df.rename(columns={
                            'date': 'Date',
                            'close': 'Close'
                        })
                        st.success("Data initialization complete!")
                        st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    df = None
        else:
            ticker = st.text_input("Enter stock symbol for analysis:", "TSLA", key="got_ticker")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Starting temporal coordinate", datetime.now() - timedelta(days=365), key="got_start")
            with col2:
                end_date = st.date_input("Ending temporal coordinate", datetime.now(), key="got_end")
            
            if st.button("Initialize Data Download"):
                with st.spinner("Establishing quantum connection..."):
                    df = fetch_stock_data(ticker, start_date, end_date)
                    if df is not None and not df.empty:
                        st.success("Data successfully retrieved from the mainframe!")
                        df.reset_index(inplace=True)
                        st.dataframe(df.head())
                    else:
                        st.error("Failed to retrieve data. Please check the ticker symbol and try again.")
                        df = None

        # Enhanced visualization section
        if df is not None and not df.empty:
            # Get numeric columns for feature selection
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if len(numeric_columns) >= 2:
                feature_cols = st.multiselect(
                    "Select features for clustering:",
                    options=numeric_columns,
                    default=numeric_columns[:2],
                    key="got_features"
                )
                
                if len(feature_cols) >= 2:
                    # Scale the features
                    scaler = StandardScaler()
                    
                    if st.button("Begin the Clustering Ritual", 
                                help="Initiate the ancient clustering ritual to unveil market houses"):
                        with st.spinner("The maesters are analyzing the patterns..."):
                            # Perform K-means clustering
                            kmeans = KMeans(n_clusters=3, random_state=42)
                            df_scaled = scaler.fit_transform(df[feature_cols])
                            clusters = kmeans.fit_predict(df_scaled)

                            # Create cluster table
                            cluster_table = df[feature_cols].copy()
                            cluster_table['House'] = clusters
                            cluster_table = cluster_table.groupby('House').mean()

                            # Create visualization
                            fig = px.scatter(
                                df,
                                x=feature_cols[0],
                                y=feature_cols[1],
                                color=clusters,
                                title="House Clustering Visualization",
                                labels={feature_cols[0]: feature_cols[0], feature_cols[1]: feature_cols[1]}
                            )

                            # Display enhanced cluster visualization
                            st.markdown("""
                                <div style="
                                    background: rgba(26,26,26,0.8);
                                    padding: 1.5rem;
                                    border-radius: 10px;
                                    border: 1px solid #D4AF37;
                                    margin: 1rem 0;
                                ">
                                    <h4 style="color: #D4AF37; text-align: center;">House Characteristics</h4>
                                </div>
                            """, unsafe_allow_html=True)

                            # Enhanced table styling
                            st.dataframe(
                                cluster_table.style.background_gradient(cmap='YlOrBr')
                                .set_properties(**{
                                    'background-color': 'rgba(26,26,26,0.8)',
                                    'color': '#D4AF37',
                                    'border': '1px solid #D4AF37'
                                }),
                                use_container_width=True
                            )

                            # Enhanced plot styling
                            fig.update_layout(
                                template='plotly_dark',
                                plot_bgcolor='rgba(26,26,26,0.8)',
                                paper_bgcolor='rgba(26,26,26,0.8)',
                                font=dict(color='#D4AF37'),
                                title=dict(
                                    font=dict(size=24, color='#D4AF37'),
                                    x=0.5,
                                    xanchor='center'
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Enhanced prediction interface
                            st.markdown("""
                                <div style="
                                    background: rgba(26,26,26,0.8);
                                    padding: 1.5rem;
                                    border-radius: 10px;
                                    border: 1px solid #D4AF37;
                                    margin: 1rem 0;
                                ">
                                    <h4 style="color: #D4AF37; text-align: center;">House Assignment Ritual</h4>
                                </div>
                            """, unsafe_allow_html=True)

                            # Input fields for prediction
                            input_values = {}
                            for feature in feature_cols:
                                min_val = float(df[feature].min())
                                max_val = float(df[feature].max())
                                mean_val = float(df[feature].mean())

                                # Adjust slider step
                                step = max((max_val - min_val) / 100, 0.01)

                                input_values[feature] = st.slider(
                                    f"Value for {feature}",
                                    min_val, max_val, mean_val,
                                    step=step,
                                    key=f"got_input_{feature}"
                                )

                            if st.button("Assign House"):
                                # Create input dataframe for prediction
                                input_df = pd.DataFrame([input_values])

                                # Scale the input
                                input_scaled = scaler.transform(input_df)

                                # Predict cluster
                                house = kmeans.predict(input_scaled)[0]

                                # Display prediction
                                st.markdown(
                                    f"<div style='background-color: rgba(212, 175, 55, 0.1); padding: 20px; border-radius: 10px; "
                                    f"border: 1px solid #D4AF37; text-align: center;'>"
                                    f"<h3 style='color: #D4AF37;'>House Assignment</h3>"
                                    f"<h2 style='color: #D4AF37; font-size: 2.5em;'>House {house}</h2>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                else:
                    st.error("You must select at least two features for the clustering analysis!")
            else:
                st.error("Not enough numeric columns available for clustering analysis!")
        else:
            st.error("Please upload a dataset first!")

        # End of the main styled container
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
