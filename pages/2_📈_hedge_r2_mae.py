import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.formula.api as smf

st.set_page_config(page_icon= "📈" ,page_title= "R2 and MAE analysis",layout="wide")

def store_value(key):
    st.session_state[key] = st.session_state["_" + key]

def load_value(key):
    if key in st.session_state:
        st.session_state["_" + key] = st.session_state[key]

# Streamlit app
st.title("Hedge R² and MAE Analysis")

if "df" not in st.session_state or "product_options" not in st.session_state:
    st.warning("Please upload the Vesper Data file to proceed.")
else:
    json_prices = st.session_state["df"]
    product_test = st.session_state["product_options"]

    # Initialize temporary widget values
    load_value("start_date")
    load_value("end_date")
    load_value("selected_product")

    # Add date filter widgets
    st.date_input(
        "Select start date",
        pd.to_datetime('01-01-2015').date(),
        key="_start_date",
        on_change=store_value,
        args=["start_date"],
    )
    st.date_input(
        "Select end date",
        pd.to_datetime(json_prices["date"].max()).date(),
        key="_end_date",
        on_change=store_value,
        args=["end_date"],
    )
    # Filter data based on selected dates
    json_prices["date"] = pd.to_datetime(json_prices["date"])
    start_date = st.session_state.get("start_date", pd.to_datetime(json_prices["date"].min()).date())
    end_date = st.session_state.get("end_date", pd.to_datetime(json_prices["date"].max()).date())
    json_prices_filtered = json_prices[
        (json_prices["date"] >= pd.to_datetime(start_date)) & (json_prices["date"] <= pd.to_datetime(end_date))
    ]

    # Selection menu for target product
    st.selectbox(
        "Select the target product (y):",
        product_test,
        key="_selected_product",
        on_change=store_value,
        args=["selected_product"],
        index=0,
    )
    selected_product = st.session_state.get("selected_product")


    # Cache the calculation results to improve performance
    @st.cache_data
    def calculate_results(json_prices_filtered, selected_product):
        # Prepare merged data
        dataframes = []

        for product in ['Butter_EEX', 'SMP, food_EEX', selected_product]:
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
        feature1 = merged_data['Butter_EEX'].values
        feature2 = merged_data['SMP, food_EEX'].values

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

    with st.container():
        columns = st.columns([2,6])   
        with columns[0]:
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

        with columns[1]:
            st.write("Select periods to plot:")
            col1, col2, col3 = st.columns(3)
            with col1:
                plot_6_months = st.checkbox("Plot 6 Months")
            with col2:
                plot_1_year = st.checkbox("Plot 1 Year")
            with col3:
                plot_2_years = st.checkbox("Plot 2 Years")


    # Function to get or calculate period results
    def get_period_results(period_key, period_start_date):
        if period_key not in st.session_state:
            json_prices_filtered_period = json_prices[
                (json_prices["date"] >= period_start_date) & (json_prices["date"] <= pd.to_datetime(end_date))
            ]
            if json_prices_filtered_period.empty:
                st.error(f"No data available for the selected period: {period_key}.")
                return None, None, None, None
            residuals_dict, results_r2, results_mae, params_dict = calculate_results(json_prices_filtered_period, selected_product)
            st.session_state[period_key] = {
                'residuals_dict': residuals_dict,
                'results_r2': results_r2,
                'results_mae': results_mae,
                'params_dict': params_dict
            }
        else:
            results = st.session_state[period_key]
            residuals_dict = results['residuals_dict']
            results_r2 = results['results_r2']
            results_mae = results['results_mae']
            params_dict = results['params_dict']
        return residuals_dict, results_r2, results_mae, params_dict

    # Plotting
    if "results_r2" in st.session_state and "results_mae" in st.session_state:
        f1_percentages = np.arange(0, 105, 5)  # 0 to 100 inclusive, steps of 5
        f1_values = f1_percentages / 100
        params_dict = st.session_state.params_dict

        fig = go.Figure()

        # Main R² Trace
        fig.add_trace(go.Scatter(
            x=f1_values,
            y=st.session_state.results_r2,
            name="R² (Selected Period)",
            line=dict(color='#023047'),
            mode='lines+markers',
            yaxis='y1',
            hovertemplate=(
                'f1: %{x:.2f}<br>' +
                'R²: %{y:.4f}<br>' +
                'c (hedge coefficient): %{customdata:.4f}'
            ),
            customdata=[params_dict[f1] for f1 in f1_percentages]
        ))

        # Main MAE Trace
        fig.add_trace(go.Scatter(
            x=f1_values,
            y=st.session_state.results_mae,
            name="MAE (Selected Period)",
            line=dict(color='#023047', dash='dot'),
            mode='lines+markers',
            yaxis='y2',
            hovertemplate=(
                'f1: %{x:.2f}<br>' +
                'MAE: %{y:.4f}<br>' +
                'Parameter: %{customdata:.4f}'
            ),
            customdata=[params_dict[f1] for f1 in f1_percentages]
        ))

        # Add lines for toggled periods
        if plot_6_months:
            period_start_date = pd.to_datetime(end_date) - pd.DateOffset(months=6)
            residuals_dict_6m, results_r2_6m, results_mae_6m, params_dict_6m = get_period_results('6_months', period_start_date)
            if results_r2_6m and results_mae_6m:
                # R² Trace for 6 Months
                fig.add_trace(go.Scatter(
                    x=f1_values,
                    y=results_r2_6m,
                    name="R² (6 Months)",
                    line=dict(color='#8ecae6'),
                    mode='lines+markers',
                    yaxis='y1',
                    hovertemplate=(
                        'f1: %{x:.2f}<br>' +
                        'R² (6 Months): %{y:.4f}<br>' +
                        'c (hedge coefficient): %{customdata:.4f}'
                    ),
                    customdata=[params_dict_6m[f1] for f1 in f1_percentages]
                ))

                # MAE Trace for 6 Months
                fig.add_trace(go.Scatter(
                    x=f1_values,
                    y=results_mae_6m,
                    name="MAE (6 Months)",
                    line=dict(color='#8ecae6', dash='dot'),
                    mode='lines+markers',
                    yaxis='y2',
                    hovertemplate=(
                        'f1: %{x:.2f}<br>' +
                        'MAE (6 Months): %{y:.4f}<br>' +
                        'Parameter: %{customdata:.4f}'
                    ),
                    customdata=[params_dict_6m[f1] for f1 in f1_percentages]
                ))

        if plot_1_year:
            period_start_date = pd.to_datetime(end_date) - pd.DateOffset(years=1)
            residuals_dict_1y, results_r2_1y, results_mae_1y, params_dict_1y = get_period_results('1_year', period_start_date)
            if results_r2_1y and results_mae_1y:
                # R² Trace for 1 Year
                fig.add_trace(go.Scatter(
                    x=f1_values,
                    y=results_r2_1y,
                    name="R² (1 Year)",
                    line=dict(color='#219ebc'),
                    mode='lines+markers',
                    yaxis='y1',
                    hovertemplate=(
                        'f1: %{x:.2f}<br>' +
                        'R² (1 Year): %{y:.4f}<br>' +
                        'c (hedge coefficient): %{customdata:.4f}'
                    ),
                    customdata=[params_dict_1y[f1] for f1 in f1_percentages]
                ))

                # MAE Trace for 1 Year
                fig.add_trace(go.Scatter(
                    x=f1_values,
                    y=results_mae_1y,
                    name="MAE (1 Year)",
                    line=dict(color='#219ebc', dash='dot'),
                    mode='lines+markers',
                    yaxis='y2',
                    hovertemplate=(
                        'f1: %{x:.2f}<br>' +
                        'MAE (1 Year): %{y:.4f}<br>' +
                        'Parameter: %{customdata:.4f}'
                    ),
                    customdata=[params_dict_1y[f1] for f1 in f1_percentages]
                ))

        if plot_2_years:
            period_start_date = pd.to_datetime(end_date) - pd.DateOffset(years=2)
            residuals_dict_2y, results_r2_2y, results_mae_2y, params_dict_2y = get_period_results('2_years', period_start_date)
            if results_r2_2y and results_mae_2y:
                # R² Trace for 2 Years
                fig.add_trace(go.Scatter(
                    x=f1_values,
                    y=results_r2_2y,
                    name="R² (2 Years)",
                    line=dict(color='#fb8500'),
                    mode='lines+markers',
                    yaxis='y1',
                    hovertemplate=(
                        'f1: %{x:.2f}<br>' +
                        'R² (2 Years): %{y:.4f}<br>' +
                        'c (hedge coefficient): %{customdata:.4f}'
                    ),
                    customdata=[params_dict_2y[f1] for f1 in f1_percentages]
                ))

                # MAE Trace for 2 Years
                fig.add_trace(go.Scatter(
                    x=f1_values,
                    y=results_mae_2y,
                    name="MAE (2 Years)",
                    line=dict(color='#fb8500', dash='dot'),
                    mode='lines+markers',
                    yaxis='y2',
                    hovertemplate=(
                        'f1: %{x:.2f}<br>' +
                        'MAE (2 Years): %{y:.4f}<br>' +
                        'Parameter: %{customdata:.4f}'
                    ),
                    customdata=[params_dict_2y[f1] for f1 in f1_percentages]
                ))

        # Layout configuration

        fig.update_layout(
            title=f"R² and MAE for {selected_product}",
            xaxis=dict(title="Weight of Butter EEX Index"),
            yaxis=dict(
                title=dict(
                    text="R²",
                    font=dict(color="blue"),  # Corrected: font instead of titlefont
                ),
                tickfont=dict(color="blue"),
            ),
            yaxis2=dict(
                title=dict(
                    text="MAE",
                    font=dict(color="red"), # Corrected: font instead of titlefont
                ),
                tickfont=dict(color="red"),
                overlaying="y",
                side="right",
            ),
            template="plotly",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.01,
            ),
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
