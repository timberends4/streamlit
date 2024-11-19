import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import statsmodels.formula.api as smf
import numpy as np

# Set Streamlit to use wide mode for better horizontal space utilization
st.set_page_config(layout="wide")

# Initialize lock states and necessary session state variables
default_states = {
    "p_1_locked": False,
    "f_1_locked": False,
    "c_1_locked": False,
    "p_2_locked": False,
    "f_2_locked": False,
    "F1": 0.5,
    "F2": 0.5,
    "C1": 1.0,
    "r_squared": "",
    "y_formula": "",
    "df": None,
    "vesper_options": [],
    "product_options": [],
    "c1_status": True,  # True means "✅", False means "❌"
    "include_intercept": True  # New flag for intercept inclusion

}

for key, default_value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Helper function to toggle lock state
def toggle_lock(key):
    st.session_state[key] = not st.session_state[key]

def toggle_intercept():
    st.session_state["include_intercept"] = not st.session_state["include_intercept"]

# Function to perform OLS regression
def run_regression():
    df = st.session_state.get('df', pd.DataFrame())
    p1_product = st.session_state.get("p_1")
    p2_product = st.session_state.get("p_2")
    y_product = st.session_state.get("y_row_1")
    start_date = st.session_state.get("start_date")
    end_date = st.session_state.get("end_date")
    include_intercept = st.session_state.get("include_intercept", True)  # New flag

    if not df.empty and all([p1_product, p2_product, y_product, start_date, end_date]):
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

        df_merged['F1'] = st.session_state.get("F1", 0.5)
        df_merged['F2'] = st.session_state.get("F2", 0.5)
        df_merged['C1'] = st.session_state.get("C1", 1.0)  # Hedge coefficient

        # Define response and independent variables
        y = df_merged[y_product].values
        P1 = df_merged[p1_product].values
        P2 = df_merged[p2_product].values

        df_reg = pd.DataFrame({
            "y_price": y,
            "term": P1 * df_merged['F1'] + P2 * df_merged['F2'],
            "P1": P1,
            "P2": P2
        })

        # Determine model formula based on c1_status, c_1_locked, and include_intercept
        c1_status = st.session_state.get("c1_status", True)
        c1_locked = st.session_state.get("c_1_locked", False)

        if not c1_status:
            # Fixed coefficient model
            if include_intercept:
                model_formula = "y_price ~ P1 + P2"
            else:
                model_formula = "y_price ~ P1 + P2 - 1"
            model = smf.ols(model_formula, data=df_reg).fit()
            y_pred = model.predict(df_reg)
            st.session_state['y_formula'] = (
                f"y = {model.params.get('P1', 0):.2f} * P1 + {model.params.get('P2', 0):.2f} * P2"
            )
            st.session_state['C1_opt'] = 0
        else:
            if c1_locked:
                # Model with C1 locked as a fixed hedge coefficient
                df_reg["y_adjusted"] = df_reg["y_price"] - df_reg["term"] * st.session_state["C1"]
                if include_intercept:
                    model_formula = "y_adjusted ~ 1"
                else: 
                    st.session_state['C1_opt'] = st.session_state['C1']
                    return st.error("Cannot run model without intercept when C1 is locked.")
                
                model = smf.ols(model_formula, data=df_reg).fit()
                y_pred = model.predict() + df_reg["term"] * st.session_state["C1"]
                if include_intercept:
                    intercept = model.params.get('Intercept', 0.0)
                    st.session_state['y_formula'] = (
                        f"y = {intercept:.2f} + "
                        f"{st.session_state['C1']:.2f} * (P1 * {st.session_state['F1']:.2f} + P2 * {st.session_state['F2']:.2f})"
                    )
                else:
                    st.session_state['y_formula'] = (
                        f"y = {st.session_state['C1']:.2f} * (P1 * {st.session_state['F1']:.2f} + P2 * {st.session_state['F2']:.2f})"
                    )
                st.session_state['C1_opt'] = st.session_state['C1']
            else:
                # Model with variable term coefficient
                model_formula = "y_price ~ term"
                if not include_intercept:
                    model_formula += " -1"
                model = smf.ols(model_formula, data=df_reg).fit()
                y_pred = model.predict(df_reg)
                term_coeff = model.params.get('term', 0.0)
                if include_intercept:
                    intercept = model.params.get('Intercept', 0.0)
                    st.session_state['y_formula'] = (
                        f"y = {intercept:.2f} + "
                        f"{term_coeff:.2f} * (P1 * {st.session_state.get('F1', 0.0)} + "
                        f"P2 * {st.session_state.get('F2', 0.0)})"
                    )
                else:
                    st.session_state['y_formula'] = (
                        f"y = {term_coeff:.2f} * (P1 * {st.session_state.get('F1', 0.0)} + "
                        f"P2 * {st.session_state.get('F2', 0.0)})"
                    )
                st.session_state['C1_opt'] = term_coeff

        # Store predictions and residuals
        st.session_state['predictions'] = pd.DataFrame({
            'date': df_merged['date'],
            'predicted_price': y_pred
        })

        # Calculate R²
        y_mean = df_reg["y_price"].mean()
        sst = ((df_reg["y_price"] - y_mean) ** 2).sum()
        ssr = ((df_reg["y_price"] - y_pred) ** 2).sum()
        st.session_state['model_results'] = model
        st.session_state['r_squared'] = 1 - (ssr / sst)

        st.success("OLS model successfully run!")


# --- UI Layout ---

st.title("OLS Hedge Effectiveness Testing")
    
index_smp = st.session_state['product_options'].index('SMP, Food_EEX_INDEX')
index_butter = st.session_state['product_options'].index('Butter_EEX_INDEX')
# Display input controls and results only if data is loaded
if st.session_state.get("df") is not None:
    product_options = st.session_state['product_options']

    # --- Row 2: Input Controls for Analysis ---
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.write("Y")
        st.selectbox("", product_options, key="y_row_1")

    with col2:
        st.write("P1")
        st.selectbox("", product_options, disabled=st.session_state.p_1_locked, key="p_1", index=index_butter)

    with col3:
        st.write("F1")
        # Avoid conflict by checking session state
        f1_value = st.session_state.get("F1", 0.5)
        st.number_input("", value=f1_value, key="F1")


    with col4:
        st.write("Start Date")
        st.date_input("", key="start_date", value=datetime(2019, 1, 1))

    with col5:
        st.write("Hedge Coefficient")
        c1_display_value = st.session_state.get("C1", 1.0)
        st.number_input("Adjustable C1", value=c1_display_value, disabled=st.session_state.c_1_locked, key="C1")

    # --- Row 3: Additional Inputs and Run Button ---
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.write("Y (Formula)")
        st.text_input("", value=st.session_state.get("y_formula", ""), key="y_row_2", disabled=True)

    with col2:
        st.write("P2")
        st.selectbox("", product_options, disabled=st.session_state.p_2_locked, key="p_2", index=index_smp)

    with col3:
        f2_value = st.session_state.get("F2", 0.5)
        st.write("F2")
        st.number_input("", value=f2_value, key="F2")

    with col4:
        st.write("End Date")
        st.date_input("", datetime.today(), key="end_date")

    with col5:
        if st.button("Run OLS Model"):
            run_regression()

    # --- Row 4: Lock Controls ---
    buttons_col1, buttons_col2, buttons_col3 = st.columns([1, 1, 1])

    with buttons_col1:
        # Toggle C1 lock/unlock button
        st.button("C currently locked: 🔒" if st.session_state.c_1_locked else "C currently unlocked: 🔓",
                on_click=toggle_lock, args=("c_1_locked",), key="c1_lock")

    with buttons_col2:
        # Toggle C include/exclude button
        st.button(
            "C currently included: ✅" if st.session_state["c1_status"] else "C currently not included: ❌",
            on_click=lambda: st.session_state.update({"c1_status": not st.session_state["c1_status"]}),
            key="toggle_emoji_button"
        )

    with buttons_col3:
        # Toggle intercept inclusion button
        st.button(
            "Intercept removed: ❌" if not st.session_state["include_intercept"] else "Intercept included: ✅",
            on_click=toggle_intercept,
            key="toggle_intercept_button"
        )
    # --- Row 5: Plot and OLS Regression Results ---

    if st.session_state.get("df") is not None:
        required_columns = {'product_sc', 'date', 'price'}
        df = st.session_state['df']

        if required_columns.issubset(df.columns):
            # Filter data for plotting
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
                actual_df = actual_df.resample('W', on='date').mean().dropna().reset_index()

                if 'date' in pred_df.columns and 'date' in actual_df.columns:
                    pred_df['date'] = pd.to_datetime(pred_df['date'])
                    actual_df['date'] = pd.to_datetime(actual_df['date'])

                    # Merge actual and predicted data
                    merged_df = pd.merge(
                        actual_df,
                        pred_df,
                        on='date',
                        how='inner',
                        suffixes=('_actual', '_pred')
                    )

                    # Check merged data
                    if not merged_df.empty:
                        merged_df['Spread'] = merged_df['price'] - merged_df['predicted_price']
                        merged_df['Spread'] = merged_df['Spread'].dropna()

                        st.session_state['Spread'] = merged_df['Spread']

                        y_actual = merged_df['price'].values
                        y_pred = merged_df['predicted_price'].values
                        pred_dates = merged_df['date']

                        # Add predicted line
                        fig.add_trace(go.Scatter(
                            x=pred_dates, y=y_pred,
                            mode='lines', name='OLS Prediction',
                            line=dict(dash='dash', color='black')
                        ))

                        # Separate Positive and Negative Spread
                        merged_df['Positive_Spread'] = merged_df['Spread'].apply(lambda x: x if x > 0 else 0)
                        merged_df['Negative_Spread'] = merged_df['Spread'].apply(lambda x: x if x < 0 else 0)

                        # Add Positive Spread Area (Green)
                        fig.add_trace(go.Scatter(
                            x=merged_df['date'],
                            y=merged_df['Positive_Spread'],
                            mode='lines',
                            name='Positive Spread',
                            line=dict(color='green'),
                            fill='tozeroy',
                            fillcolor='rgba(0,255,0,0.5)',
                            showlegend=True
                        ))

                        # Add Negative Spread Area (Red)
                        fig.add_trace(go.Scatter(
                            x=merged_df['date'],
                            y=merged_df['Negative_Spread'],
                            mode='lines',
                            name='Negative Spread',
                            line=dict(color='red'),
                            fill='tozeroy',
                            fillcolor='rgba(255,0,0,0.5)',
                            showlegend=True
                        ))
                    else:
                        st.warning("Merged data for actual and predicted values is empty.")

            # Add vertical lines for start and end dates
            fig.add_vline(x=st.session_state.get("start_date"), line=dict(color='red', width=2, dash='dash'))
            fig.add_vline(x=st.session_state.get("end_date"), line=dict(color='red', width=2, dash='dash'))

            st.plotly_chart(fig, use_container_width=True)


    # --- Additional Section: Price Calculation and Residual Distribution ---
    col1, col2 = st.columns(2)
    with col1:
        st.write("Price Calculation")

        if st.session_state.get("C1_opt") is not None:
            p1_value = st.number_input("Enter P1 value:", value=0.0)
            p2_value = st.number_input("Enter P2 value:", value=0.0)

            intercept = st.session_state['model_results'].params.get("Intercept", 0.0)
            c1_opt = st.session_state.get("C1_opt", 1.0)
            f1 = st.session_state.get("F1", 0.5)
            f2 = st.session_state.get("F2", 0.5)
            
            params = st.session_state['model_results'].params

            if st.session_state.get("c1_status"):
                # Calculate the back price based on the model formula
                back_price = intercept + c1_opt * (p1_value * f1 + p2_value * f2)

                st.write("Calculated Back Price:", back_price)
                # Add debug statements
                st.write("Back Price Calculation Variables:")
                st.write(f"Intercept: {intercept}")
                st.write(f"c1_opt: {c1_opt}")
                st.write(f"p1_value: {p1_value}, p2_value: {p2_value}")
                st.write(f"f1: {f1}, f2: {f2}")
            else:
                # Using model y_price ~ P1 + P2 -1
                f1_coeff = params.get('P1', 0.0)
                f2_coeff = params.get('P2', 0.0)
                back_price = intercept + p1_value * f1_coeff + p2_value * f2_coeff

                st.write("Calculated Back Price:", back_price)
                # Add debug statements
                st.write("Back Price Calculation Variables:")
                st.write(f"Intercept: {intercept}")
                st.write(f"p1_value: {p1_value}, p2_value: {p2_value}")
                st.write(f"f1: {f1_coeff}, f2: {f2_coeff}")

        else:
            st.warning("Run the OLS model to calculate the back price.")

    with col2:
        st.write("Fit Results")
        if st.session_state.get("model_results") is not None:
            st.write(f"Custom R²: {st.session_state['r_squared']:.4f}")
            st.write(st.session_state['model_results'].summary())

        if 'Spread' in st.session_state:
            fig = px.histogram(st.session_state['Spread'], nbins=50, title='Distribution of Model Residuals')
            fig.update_layout(xaxis_title='Residual (Spread)', yaxis_title='Frequency')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Run the model first to calculate residuals.")

else:
    st.warning("Please upload the Vesper data file to proceed.")