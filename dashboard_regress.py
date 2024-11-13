import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import statsmodels.formula.api as smf
import numpy as np
from streamlit.web import cli as stcli
import os

# Set Streamlit to use wide mode for better horizontal space utilization
st.set_page_config(layout="wide")
# Initialize lock states and necessary session state variables
for key, default_value in {
    "p_1_locked": False, 
    "f_1_locked": False, 
    "c_1_locked": False, 
    "p_2_locked": False, 
    "f_2_locked": False, 
    "F1": 0.0,
    "F2": 0.0,
    "C1": 1.0,
    "r_squared": "", 
    "y_formula": "", 
    "df": None, 
    "vesper_options": [],
    "product_options": []}.items():

    if key not in st.session_state:
        st.session_state[key] = default_value

# Helper function to toggle lock state
def toggle_lock(key):
    st.session_state[key] = not st.session_state[key]

# Function to perform OLS regression with bootstrapped confidence intervals
def run_regression():
    # Load data from session state
    df = st.session_state.get('df', pd.DataFrame())
    p1_product = st.session_state.get("p_1")
    p2_product = st.session_state.get("p_2")
    y_product = st.session_state.get("y_row_1")
    start_date = st.session_state.get("start_date")
    end_date = st.session_state.get("end_date")
    
    if not df.empty and p1_product and p2_product and y_product and start_date and end_date:
        # Filter by date range and relevant products
        df_filtered = df[
            (df['product_sc'].isin([p1_product, p2_product, y_product])) &
            (df['date'] >= pd.to_datetime(start_date)) & 
            (df['date'] <= pd.to_datetime(end_date))
        ]

        df_merged = df_filtered.pivot(index='date', columns='product_sc', values='price').dropna().reset_index()

        df_merged = (
            df_merged.resample('W', on='date')[df_merged.select_dtypes(include='number').columns]
            .mean()
            .dropna()
            .reset_index()
        )

        df_merged['F1'] = st.session_state.get("F1", 0.0) 
        df_merged['F2'] = st.session_state.get("F2", 0.0)  
        df_merged['C1'] = st.session_state.get("C1", 1.0) # Hedge coefficient
        # Define response and independent variables
        y = df_merged[y_product].values
        P1 = df_merged[p1_product].values
        P2 = df_merged[p2_product].values
        F1 = df_merged['F1'].values
        F2 = df_merged['F2'].values

        df = pd.DataFrame({
            "y_price": y,
            "term": P1*F1 + P2*F2,
            "P1": P1,
            "P2": P2
        })
        
        if st.session_state.get("c1_status") == False:
            # Fixed coefficient model
            model = smf.ols("y_price ~ P1 + P2 - 1", data=df)
            result = model.fit()
            y_pred = result.predict(df)
            st.session_state['y_formula'] = (
                f"y = {result.params.get('Intercept', 0):.2f} + "
                f"{result.params.get('P1', 0):.2f} * P1 + {result.params.get('P2', 0):.2f} * P2"
            )

        else:
            if st.session_state.get("c_1_locked"):
                # Model with C1 locked as a fixed hedge coefficient
                df["y_adjusted"] = df["y_price"] - df["term"] * st.session_state["C1"]
                model = smf.ols("y_adjusted ~ 1", data=df)
                result = model.fit()
                y_pred = result.predict() + df["term"] * st.session_state["C1"]
                st.session_state['y_formula'] = (
                    f"y = {result.params.get('Intercept', 0):.2f} + "
                    f"{st.session_state['C1']:.2f} * (P1 * {st.session_state['F1']:.2f} + P2 * {st.session_state['F2']:.2f})"
                )
                st.session_state['C1_opt'] = st.session_state['C1']

            else:
                # Model with variable term coefficient
                model = smf.ols("y_price ~ term", data=df)
                result = model.fit()
                y_pred = result.predict(df)
                st.session_state['y_formula'] = (
                    f"y = {result.params.get('Intercept', 0):.2f} + "
                    f"{result.params.get('term', 0):.2f} * (P1 * {st.session_state.get('F1', 0.0)} + "
                    f"P2 * {st.session_state.get('F2', 0.0)})"
                )
                st.session_state['C1_opt'] = result.params.get('term', 0)

        # Store predictions and residuals
        st.session_state['predictions'] = pd.DataFrame({
            'date': df_merged['date'],
            'predicted_price': y_pred   
        })
        
        # Calculate Total Sum of Squares (SST) and Residual Sum of Squares (SSR)
        y_mean = df["y_price"].mean()
        sst = ((df["y_price"] - y_mean) ** 2).sum()
        ssr = ((df["y_price"] - y_pred) ** 2).sum()
        
        # Store model results and custom R² calculation in session state
        st.session_state['model_results'] = result
        st.session_state['r_squared'] = 1 - (ssr / sst)

        st.success("OLS model successfully run!")

# Row 1: File Upload and Product Options
with st.container():
    st.title("OLS Hedge Effectiveness Testing")
    uploaded_file = st.file_uploader("Upload a file for analysis", type=["csv", "xlsx", "json"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".json"):
                df = pd.read_json(uploaded_file)
            st.write("File successfully uploaded and read!")

            # Ensure 'product_sc' is created only if 'product' and 'source' columns are present
            if 'product' in df.columns and 'source' in df.columns:
                df['product_sc'] = df['product'] + " (" + df['source'] + ")"
                vesper_options = df['product_sc'].unique().tolist()
                vesper_options = [x for x in vesper_options if 'EEX' in x]
                st.session_state['vesper_options'] = vesper_options

                # Create the product options list filtered by 'EEX' for the selection list only
                product_options = df[df['source'] == 'EEX']['product_sc'].unique().tolist()
                st.session_state['product_options'] = vesper_options

                # Store the full DataFrame without filtering for use in analysis
                st.session_state['df'] = df
            else:
                st.warning("The uploaded file must contain 'product' and 'source' columns.")
        
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

# Display input controls and results only if data is loaded
if st.session_state.get("df") is not None:

    product_options = st.session_state['product_options']
    
    # Row 2: Input Controls for Analysis (without automatic on_change triggers)
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.write("Y")
            st.selectbox("", st.session_state['vesper_options'], key="y_row_1")

        with col2:
            st.write("P1")
            st.selectbox("", product_options, disabled=st.session_state.p_1_locked, key="p_1")

        with col3:
            st.write("F1")
            f1 = st.session_state.get("F1", 0.0)
            st.number_input("", value=f1, key="F1")

        with col4:
            st.write("Start Date")
            st.date_input("", datetime.today(), key="start_date")

        with col5:
            st.write("Hedge Coefficient")
            
            # Display the optimized C1_opt if it exists, otherwise show the user-defined C1 as the default.
            if st.session_state.get("C1_opt") is not None:
                # Display optimized C1 separately as a static value
                st.text(f"Optimized C1: {st.session_state['C1_opt']:.2f}")
                c1_display_value = st.session_state.get("C1", 1.0)  # Display user-adjustable C1 value in the input box
            else:
                c1_display_value = st.session_state.get("C1", 1.0)
            
            # Input box for the user-defined hedge coefficient C1
            st.number_input("Adjustable C1", value=c1_display_value, disabled=st.session_state.c_1_locked, key="C1")

            button_col1, button_col2 = st.columns(2)
            with button_col1:
                st.button("🔒" if st.session_state.c_1_locked else "🔓", on_click=toggle_lock, args=("c_1_locked",), key="c1_lock")
            
            with button_col2:
                # Initialize the status in session state if it doesn't already exist
                if "c1_status" not in st.session_state:
                    st.session_state["c1_status"] = True  # False means "❌", True means "✅"

                # Define a function to toggle the emoji status
                def toggle_status():
                    st.session_state["c1_status"] = not st.session_state["c1_status"]

                # Display the button with the appropriate emoji
                emoji = "✅" if st.session_state["c1_status"] else "❌"
                st.button(emoji, key="toggle_emoji_button", on_click=toggle_status)

    # Row 3: Additional Inputs and Run Button
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.write("Y (Formula)")
            st.text_input("", value=st.session_state.get("y_formula", ""), key="y_row_2", disabled=True)

        with col2:
            st.write("P2")
            st.selectbox("", product_options, disabled=st.session_state.p_2_locked, key="p_2")

        with col3:
            st.write("F2")
            f2 = st.session_state.get("F2", 0.0)
            st.number_input("", value=f2, key="F2")

        with col4:
            st.write("End Date")
            st.date_input("", datetime.today(), key="end_date")

        with col5:
            # Run OLS model button
            if st.button("Run OLS Model"):
                run_regression()

    # Row 4: Plot and OLS Regression Results
    with st.container():
        col1, col2 = st.columns([0.7, 0.3])

        with col1:
            st.write("Plot")
            
            if uploaded_file is not None:
                required_columns = {'product_sc', 'date', 'price'}
                if required_columns.issubset(df.columns):
                    # Filter data for the selected period and products
                    df_plot = df[df['product_sc'].isin([
                        st.session_state.get("p_1"),
                        st.session_state.get("p_2"),
                        st.session_state.get("y_row_1")
                    ])]

                    fig = px.line(
                        df_plot,
                        x='date',
                        y='price',
                        color='product_sc',
                        title="Product Comparison with OLS Prediction",
                        labels={"date": "Date", "price": "Price", "product_sc": "Product"}
                    )

                    # Add OLS predictions
                    if 'predictions' in st.session_state:
                        pred_df = st.session_state['predictions']
                        actual_df = df_plot[df_plot['product_sc'] == st.session_state.get("y_row_1")][['date', 'price']]
                        
                        if 'date' in pred_df.columns and 'date' in actual_df.columns:

                            # Convert date columns to datetime
                            pred_df['date'] = pd.to_datetime(pred_df['date'])
                            actual_df['date'] = pd.to_datetime(actual_df['date'])
                            
                            # Set 'date' as the index to allow resampling
                            pred_df.set_index('date', inplace=True)
                            actual_df.set_index('date', inplace=True)
                            
                            # Resample to Wednesdays using mean aggregation for simplicity
                            pred_df_resampled = pred_df.resample('W-WED').mean()
                            actual_df_resampled = actual_df.resample('W-WED').mean()
                            
                            # Reset index to turn 'date' back into a column for merging
                            pred_df_resampled.reset_index(inplace=True)
                            actual_df_resampled.reset_index(inplace=True)
                            
                            # Merge on 'date' to align the resampled data
                            merged_df = pd.merge(actual_df_resampled, pred_df_resampled, on='date', how='inner', suffixes=('_actual', '_pred'))
                            merged_df['Spread'] = merged_df['price'] - merged_df['predicted_price']

                            y_actual = merged_df['price'].values
                            y_pred = merged_df['predicted_price'].values
                            pred_dates = merged_df['date']

                            # Calculate the spread (difference) between actual y and predicted y
                            spread = merged_df['Spread'].values

                            st.session_state['Spread'] = spread
                            # Plot actual vs. predicted lines
                            fig.add_trace(go.Scatter(
                                x=pred_dates, y=y_pred,
                                mode='lines', name='OLS Prediction',
                                line=dict(dash='dash', color='black')
                            ))

                            # Separate indices for positive (green) and negative (red) spread
                            indices_green = spread <= 0
                            indices_red = spread > 0

                            # Add green shaded areas for positive spread
                            fig.add_trace(go.Scatter(
                                x=pred_dates[indices_green], 
                                y=spread[indices_green],  # Use the spread where it's positive, else 0
                                fill='tozeroy', mode='none', showlegend=False,
                                fillcolor='green'  # Green for positive spread
                            ))

                            # Add red shaded areas for negative spread
                            fig.add_trace(go.Scatter(
                                x=pred_dates[indices_red], 
                                y=spread[indices_red],  # Use the spread where it's negative, else 0
                                fill='tozeroy', mode='none', showlegend=False,
                                fillcolor='red'  # Red for negative spread
                            ))

                    # Add vertical lines for start and end dates
                    fig.add_vline(x=st.session_state.get("start_date"), line=dict(color='red', width=2, dash='dash'))
                    fig.add_vline(x=st.session_state.get("end_date"), line=dict(color='red', width=2, dash='dash'))

                    st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("Fit Results")

            if st.session_state.get("model_results") is not None:
                # if st.session_state.get("c_1_locked"):
                st.write("C1 is locked. Custom R²: ", st.session_state['r_squared'])
                st.write(st.session_state['model_results'].summary(slim=True))   

    with st.container():
        col1, col2, col3 = st.columns([0.4, 0.3, 0.4])
        with col1:
            st.write("Price Calculation")

            # Ensure that we have an optimized model before allowing back price calculation
            if st.session_state.get("C1_opt") is not None:
                # Input boxes for P1 and P2 values
                p1_value = st.number_input("Enter P1 value:", value=0.0)
                p2_value = st.number_input("Enter P2 value:", value=0.0)

                # Retrieve optimized model coefficients from session state
                intercept = st.session_state['model_results'].params.get("Intercept", 0.0)
                c1_opt = st.session_state.get("C1_opt", 1.0)  # Optimized C1
                f1 = st.session_state.get("F1", 0.0)  # Fixed F1
                f2 = st.session_state.get("F2", 0.0)  # Fixed F2

                if st.session_state.get("c1_status") == False:    
                    # Calculate the back price based on the model formula
                    back_price = intercept + c1_opt * (p1_value * f1 + p2_value * f2)
                else:
                    f1 = st.session_state.get("model_results").params.get("P1", 0.0)
                    f2 = st.session_state.get("model_results").params.get("P2", 0.0)
                    back_price = intercept + p1_value * f1 + p2_value * f2

                # Display the calculated back price
                st.write("Calculated Back Price:", back_price)
            else:
                st.warning("Run the OLS model to calculate the back price.")

        with col2:
            # Check if spread values are available in merged_df
            if 'Spread' in st.session_state:
                # Plot distribution of spread (error) values
                fig = px.histogram(st.session_state['Spread'], nbins=150, title='Distribution of Model Errors (Residuals)')
                fig.update_layout(xaxis_title='Error (Spread)', yaxis_title='Frequency')

                # Show the histogram plot
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Run the model first to calculate residuals.")