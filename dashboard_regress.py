import streamlit as st
import pandas as pd

import streamlit as st
from streamlit import session_state



def main():
    # Initialize shared state variables
    default_states = {
        "df": None,
        "vesper_options": [],
        "residuals_dict": {},
        "results_r2": None,
        "results_mae": None,
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
        "c1_status": True,  # True means "✅", False means "❌"
        "include_intercept": True,  # Flag for intercept inclusion
        "start_date": None,
        "end_date": None,
        "y_row_1": "Butter_EEX_INDEX",
        "p_1": None,
        "p_2": None,
    }

    if "df" not in st.session_state:
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    st.title("Upload the Vesper file for analysis")

    # Row 1: File Upload and Product Options
    with st.container():
        if 'df' in st.session_state and st.session_state['df'] is not None:

            st.write("File successfully uploaded and read!")
        else:
                uploaded_file = st.file_uploader("Upload a file for analysis", type=["json"])
                
                if uploaded_file is not None:
                    try:
                        df = pd.read_json(uploaded_file)

                        # Ensure 'product_sc' is created only if 'product' and 'source' columns are present
                        if 'product' in df.columns and 'source' in df.columns:
                            df['product_sc'] = df['product'] + '_' + df['source']
                            vesper_options = df['product_sc'].unique().tolist()
                            vesper_options = [x for x in vesper_options if (('EEX_INDEX' in x) or ('Cagliata_EU_VPI' in x) or ('Emmental_EU_VPI' in x))] 
                            st.session_state['vesper_options'] = vesper_options

                            # Create the product options list
                            st.session_state['product_options'] = vesper_options

                            df = df.drop_duplicates(subset=['date', 'product_sc']).reset_index(drop=True)
                            # Store the DataFrame for use in analysis
                            st.session_state['df'] = df

                            # Adjust 'y_row_1', 'p_1', and 'p_2' to valid options
                            if st.session_state.get('y_row_1') not in st.session_state['product_options']:
                                st.session_state['y_row_1'] = st.session_state['product_options'][0]

                            if st.session_state.get('p_1') not in st.session_state['product_options']:
                                st.session_state['p_1'] = st.session_state['product_options'][0]

                            if st.session_state.get('p_2') not in st.session_state['product_options']:
                                st.session_state['p_2'] = st.session_state['product_options'][0]

                            st.session_state['start_date'] = df['date'].min()
                            st.session_state['end_date'] = df['date'].max()
                            
                            st.write("File successfully uploaded and read!")
                        else:
                            st.warning("The uploaded file must contain 'product' and 'source' columns.")

                    except Exception as e:
                        st.error(f"An error occurred while reading the file: {e}")
                else:
                    st.write("Please upload a file for analysis.")

if __name__ == "__main__":
    main()
