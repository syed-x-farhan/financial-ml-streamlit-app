# Financial Machine Learning App

A multi-themed financial machine learning application built with Streamlit. This interactive app features three unique themes, each with its own machine learning model for financial analysis.

## Features

- **Welcome Page**: Introduction and navigation hub
- **Zombie Theme**: Linear Regression analysis with a spooky twist
- **Futuristic Theme**: Logistic Regression with a sci-fi interface
- **Game of Thrones Theme**: K-Means Clustering with Westeros-inspired visuals

## Setup Instructions

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Data Sources

- Upload your own CSV files (Kragle dataset)
- Fetch real-time data using Yahoo Finance (yfinance)

## Requirements

- Python 3.8+
- See requirements.txt for full list of dependencies
