import streamlit as st
from del_p import custom_training
from page1_eda import custom_eda
from descriptions import upsampling_description, model_description

st.set_page_config(layout="wide", page_title="Team 4's Project")


def intro():
    st.write("# Welcome to Team 4's Nurse Attrition Analysis Project 🏥")
    st.sidebar.success("Select a section to begin.")

    st.markdown("""
    Welcome to the interactive component of Team 4's Nurse Attrition Analysis project, aimed at understanding and addressing nurse turnover in the U.S. healthcare system. This tool allows you to explore how different models and techniques can predict and potentially prevent nurse attrition. The following features are included:

    1. **Data Exploration**: Dive into the synthetic dataset tailored to simulate realistic scenarios of nurse attrition in healthcare. Understand key variables like job satisfaction, workload, and career opportunities that correlate with attrition rates.
    2. **Upsampling Techniques Overview**: Examine methods such as Random Over and Under Sampling, SMOTE, and CGANs to balance the dataset, ensuring fair representation across all classes.
    3. **Model Overview**: Discover how various predictive models, including Logistic Regression, Random Forest, SVM, KNN, and XGBoost, are used to analyze factors influencing nurse retention.
    4. **Model Training Simulator**: Test different models, thresholds, and upsampling techniques to see their effects on prediction outcomes, offering a hands-on experience with the tools used in our research.

    **Project Context**
    The high rate of nurse turnover has significant implications for patient care quality and operational efficiency in healthcare facilities. With nearly 4 million nurses in the U.S. and an anticipated need for over 275,000 more by 2030, understanding and curbing nurse attrition is vital. Our project, backed by a comprehensive analysis of various factors that contribute to nurse turnover, seeks to provide actionable insights for healthcare providers to develop effective retention strategies, thus improving outcomes and reducing costs associated with nurse turnover.
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
    "Home": intro,
    "Exploratory Data Analysis": page_1_eda,
    "Upsampling Explained": page_2_upsampling,
    "Model Description": page_3_modeling,
    "Train your own Model": page_4_train,
}

if 'current_demo' not in st.session_state:
    st.session_state['current_demo'] = None

# Sidebar selection box
demo_name = st.sidebar.selectbox("Choose Practice Type", list(page_names_to_funcs.keys()), index=0)

# Dynamically call the selected demo function
if demo_name in page_names_to_funcs:
    page_names_to_funcs[demo_name]()
