import streamlit as st

def upsampling_description():
    st.title("Upsampling Techniques")
    st.markdown("""
Explore the various upsampling techniques used to address class imbalance, which equips our models with a more representative training dataset:

- **Random Over-Sampling**: This technique duplicates instances from the minority class, such as attrition events, to even out the class distribution. While straightforward, it risks overfitting since the additional instances are just copies and don’t provide new information.

- **Random Under-Sampling**: This method reduces the number of instances in the majority class by removing them randomly. It balances the class distribution but can lead to the loss of potentially valuable data, as it indiscriminately discards instances.

- **SMOTE (Synthetic Minority Over-sampling Technique)**: SMOTE creates new, synthetic instances by interpolating between existing instances in the minority class and their nearest neighbors. This enhances the diversity of the minority class with new, unique examples, thereby decreasing the risk of overfitting.

- **Conditional Generative Adversarial Networks (cGANs)**: cGANs use two models—a generator and a discriminator—that work against each other to produce new, synthetic instances of the minority class. The generator creates new data points while the discriminator evaluates their authenticity. This method not only generates diverse and realistic samples but also helps in significantly enhancing the training dataset without repeating existing instances.

These techniques are crucial for ensuring our predictive models are trained on balanced data, improving their accuracy and generalizability.
""")

def model_description():
    st.title("Model Descriptions")
    st.markdown("""
    Explore the predictive models employed in our analysis, each handpicked for its unique strengths and ability to tackle specific aspects of the data:

    - **Logistic Regression**: A robust statistical model that excels at binary classification problems, estimating the probability of an outcome based on input features.

    - **Random Forest**: An ensemble powerhouse that constructs multiple decision trees and combines their predictions for enhanced accuracy in classification, regression, and beyond.

    - **Support Vector Machine (SVM)**: A versatile and powerful classifier that ingeniously finds the optimal hyperplane to separate data classes with maximum margin.

    - **K-Nearest Neighbors (KNN)**: An intuitive, non-parametric approach that classifies samples based on the majority vote of their nearest neighbors, leveraging similarity for prediction.

    - **XGBoost**: A high-performance implementation of gradient boosted decision trees, renowned for its speed and precision, often at the forefront of data science competitions.
    """)
