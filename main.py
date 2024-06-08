import streamlit as st
from utilities.sidebar import add_sidebar
from utilities.visualization import get_radar_chart
from utilities.prediction import add_predictions

def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    input_data = add_sidebar()
    
    st.title("Breast Cancer Predictor")
    st.write(
        "This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. "
        "You can also update the measurements by hand using the sliders in the sidebar."
    )

    # Use responsive columns
    col1, col2 = st.columns([3, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    
    with col2:
        add_predictions(input_data)
        
if __name__ == "__main__":
    main()
