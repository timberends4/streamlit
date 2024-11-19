import streamlit as st
import pandas as pd


def main():
    st.title("Upload the Vesper file for analysis")

    # Row 1: File Upload and Product Options
    with st.container():
        uploaded_file = st.file_uploader("Upload a file for analysis", type=["json"])

        if uploaded_file is not None:
            try:
                df = pd.read_json(uploaded_file)

                # Ensure 'product_sc' is created only if 'product' and 'source' columns are present
                if 'product' in df.columns and 'source' in df.columns:
                    df['product_sc'] = df['product'] + '_' + df['source']
                    vesper_options = df['product_sc'].unique().tolist()
                    vesper_options = [x for x in vesper_options if (('EEX' in x) or ('Cagliata_EU_VPI' in x) or ('Emmental_EU_VPI' in x))] 
                    st.session_state['vesper_options'] = vesper_options

                    # Create the product options list filtered by 'EEX' for the selection list only
                    # product_options = df[df['source'] == 'EEX_INDEX']['product_sc'].unique().tolist() 
                    st.session_state['product_options'] = vesper_options

                    df = df.drop_duplicates(subset=['date', 'product_sc']).reset_index(drop=True)
                    # Store the full DataFrame without filtering for use in analysis
                    st.session_state['df'] = df
                    
                    st.write("File successfully uploaded and read!")
                else:
                    st.warning("The uploaded file must contain 'product' and 'source' columns.")
            
            except Exception as e:
                st.error(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    main()
