import streamlit as st
from del_p import custom_training
from page1_eda import custom_eda
from descriptions import upsampling_description, model_description

st.set_page_config(layout="wide", page_title="Team 4's Project", page_icon=":pencil:")


def intro():
    st.write("# Welcome to Team 4's Project! 👋")
    st.sidebar.success("Select a section begin.")

    st.markdown("""
    Welcome to the interactive component of Team 4's Nurse Attrition Analysis project! This tool is crafted to assist in exploring various aspects of model training and the impact of different techniques on predicting nurse attrition. The following features are included:

    1. **Data Exploration**: Delve into the synthetic dataset designed to reflect realistic nurse attrition scenarios in healthcare. Use this module to understand the data's characteristics and how they correlate with attrition rates.
    2. **Upsampling Techniques Overview**: Explore different methods such as Random Over and Under Sampling, SMOTE (Synthetic Minority Over-sampling Technique), and CGANs (Conditional Generative Adversarial Networks) to balance the dataset.
    3. **Model Overview**: Learn about various models used, including Logistic Regression, Random Forest, SVM (Support Vector Machine), KNN (K-Nearest Neighbors), and XGBoost.
    4. **Model Training Simulator**: Experiment with different models, thresholds, and upsampling techniques to see how they influence prediction outcomes. This feature allows you to adjust parameters and instantly visualize the effects.

    """)



def page_1_eda():
    custom_eda()

def page_2_upsampling():
    upsampling_description()

def page_3_modeling():
    model_description()

def page_4_train():
    custom_training()


page_names_to_funcs = {
    "—": intro,
    "EDA": page_1_eda,
    "Upsampling Explained": page_1_eda,
    "Model description": page_3_modeling,
    "Train your own Model": page_4_train,
}

if 'current_demo' not in st.session_state:
    st.session_state['current_demo'] = None

# Sidebar selection box
demo_name = st.sidebar.selectbox("Choose Practice Type", list(page_names_to_funcs.keys()), index=0)

# Dynamically call the selected demo function
if demo_name in page_names_to_funcs:
    page_names_to_funcs[demo_name]()
