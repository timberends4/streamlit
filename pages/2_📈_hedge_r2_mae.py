import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.formula.api as smf

st.set_page_config(page_icon= "📈" ,page_title= "R2 and MAE analysis",layout="wide")

# Streamlit app
st.title("Hedge R² and MAE Analysis")

# Initialize session state for residuals
if "residuals_dict" not in st.session_state:
    st.session_state.residuals_dict = None
if "results_r2" not in st.session_state:
    st.session_state.results_r2 = None
if "results_mae" not in st.session_state:
    st.session_state.results_mae = None
if "params_dict" not in st.session_state:
    st.session_state.params_dict = None
if "df" not in st.session_state:
    st.session_state.df = None

if st.session_state["df"] is None or "product_options" not in st.session_state:
    st.warning("Please upload the Vesper Data file to proceed.")
else:
    json_prices = st.session_state['df']
    product_test = st.session_state['product_options']

    # Add date filter widgets
    start_date = st.date_input("Select start date", pd.to_datetime(json_prices['date'].min()).date())
    end_date = st.date_input("Select end date", pd.to_datetime(json_prices['date'].max()).date())

    # Filter data based on selected dates
    json_prices['date'] = pd.to_datetime(json_prices['date'])
    json_prices_filtered = json_prices[(json_prices['date'] >= pd.to_datetime(start_date)) & (json_prices['date'] <= pd.to_datetime(end_date))]

    # Selection menu for target product
    selected_product = st.selectbox("Select the target product (y):", product_test)

    # Cache the calculation results to improve performance
    @st.cache_data
    def calculate_results(json_prices_filtered, selected_product):
        # Prepare merged data
        dataframes = []

        for product in ['Butter_EEX_INDEX', 'SMP, food_EEX_INDEX', selected_product]:
            product_data = json_prices_filtered[json_prices_filtered['product_sc'] == product]
            if product_data.empty:
                raise ValueError(f"No data available for product: {product}")
            product_data = product_data.drop_duplicates(subset='date').set_index('date')['price'].rename(product)
            dataframes.append(product_data)

        merged_data = pd.concat(dataframes, axis=1, join='inner').dropna()
        
        # Define variables and precompute
        f1_percentages = np.arange(0, 105, 5)  # 0 to 100 inclusive, steps of 5
        f1_values = f1_percentages / 100
        f2_values = 1 - f1_values
        y_price = merged_data[selected_product].values
        feature1 = merged_data['Butter_EEX_INDEX'].values
        feature2 = merged_data['SMP, food_EEX_INDEX'].values

        results_r2 = []
        results_mae = []
        residuals_dict = {}
        params_dict = {}  # Dictionary to store estimated parameters

        for f1_percentage, f1, f2 in zip(f1_percentages, f1_values, f2_values):
            weighted_feature = f1 * feature1 + f2 * feature2

            # Fit model
            df = pd.DataFrame({'y_price': y_price, 'weighted_feature': weighted_feature})
            model = smf.ols("y_price ~ weighted_feature", data=df)
            result = model.fit()

            # Store results using integer keys
            results_r2.append(result.rsquared)
            results_mae.append(np.mean(np.abs(result.resid)))
            residuals_dict[f1_percentage] = result.resid  # Store residuals in dictionary
            params_dict[f1_percentage] = result.params['weighted_feature']  # Store parameter

        return residuals_dict, results_r2, results_mae, params_dict

    # Button to trigger calculations
    if st.button("Calculate"):
        try:
            residuals_dict, results_r2, results_mae, params_dict = calculate_results(json_prices_filtered, selected_product)
            st.session_state.residuals_dict = residuals_dict
            st.session_state.results_r2 = results_r2
            st.session_state.results_mae = results_mae
            st.session_state.params_dict = params_dict
            st.success("Calculations completed!")
        except ValueError as e:
            st.error(str(e))

    if st.session_state.results_r2 and st.session_state.results_mae:
        f1_percentages = np.arange(0, 105, 5)  # 0 to 100 inclusive, steps of 5
        f1_values = f1_percentages / 100
        params_dict = st.session_state.params_dict

        fig = go.Figure()

        # R² Trace
        fig.add_trace(go.Scatter(
            x=f1_values,
            y=st.session_state.results_r2,
            name="R²",
            line=dict(color='blue'),
            mode='lines+markers',
            yaxis='y1',
            hovertemplate=(
                'f1: %{x:.2f}<br>' +
                'R²: %{y:.4f}<br>' +
                'c (hedge coefficient): %{customdata:.4f}'
            ),
            customdata=[params_dict[f1] for f1 in f1_percentages]  # Add parameter values as custom data
        ))

        # MAE Trace
        fig.add_trace(go.Scatter(
            x=f1_values,
            y=st.session_state.results_mae,
            name="MAE",
            line=dict(color='red', dash='dot'),
            mode='lines+markers',
            yaxis='y2',
            hovertemplate=(
                'f1: %{x:.2f}<br>' +
                'MAE: %{y:.4f}<br>' +
                'Parameter: %{customdata:.4f}'
            ),
            customdata=[params_dict[f1] for f1 in f1_percentages]  # Add parameter values as custom data
        ))

        # Layout configuration
        fig.update_layout(
            title=f"R² and MAE for {selected_product}",
            xaxis=dict(title="Ratio Butter_EEX_INDEX"),
            yaxis=dict(title="R²", titlefont=dict(color="blue"), tickfont=dict(color="blue")),
            yaxis2=dict(title="MAE", titlefont=dict(color="red"), tickfont=dict(color="red"),
                        overlaying="y", side="right"),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Display the plot
        st.plotly_chart(fig)

        # Slider for error distribution
        st.write("Error Distribution for selected f1 value")
        selected_f1_percentage = st.slider("Select f1 percentage for Error Distribution", min_value=0, max_value=100, step=5)
        selected_f1 = selected_f1_percentage / 100

        if selected_f1_percentage in st.session_state.residuals_dict:
            selected_residuals = st.session_state.residuals_dict[selected_f1_percentage]

            # Define bin edges for fixed range and equal widths
            bin_edges = np.arange(-1000, 1050, 50)  # Bins from -1000 to 1000 with step size 50

            # Create histogram data
            hist_values, _ = np.histogram(selected_residuals, bins=bin_edges)

            # Create DataFrame for Streamlit bar chart
            bin_centers = bin_edges[:-1] + 25  # Calculate bin centers for labeling
            histogram_df = pd.DataFrame({
                "Residuals Frequency": hist_values,
                "Bins": bin_centers  # Use bin centers for labels
            })

            # Plot histogram using Streamlit bar chart
            st.bar_chart(histogram_df.set_index("Bins"))

            # Add descriptive text
            st.write(f"Error Distribution for f1 = {selected_f1:.2f}, f2 = {1 - selected_f1:.2f}")
            st.write("Residuals represent the difference between actual and predicted values.")
