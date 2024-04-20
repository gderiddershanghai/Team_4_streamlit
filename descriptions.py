import streamlit as st

def upsampling_description():
    st.title("Upsampling Techniques")
    st.markdown("""
    In our nurse attrition project, it was important to address the imbalance in our dataset. Nurses leaving their jobs happens far less often than them staying, which can lead models to miss these important events. To overcome this, we use advanced upsampling techniques to help our models accurately reflect the realities of job attrition, avoiding bias towards the more common outcome.

    **Enhancing Predictive Accuracy with Upsampling**
    Traditional models typically struggle with rare events, usually predicting the most common scenario. We use upsampling to balance the dataset, improving our models’ ability to learn from the rarer cases of nurses quitting. This method boosts predictive accuracy, helping our models identify key patterns of attrition. We use the following upsampling techniques:""")

    # Random Over-Sampling
    st.subheader("Random Over-Sampling")
    st.image("/Data/random_upsampling.png", caption="Random Over-Sampling")
    st.write("This technique duplicates instances from the minority class, such as attrition events, to even out the class distribution. While straightforward, it risks overfitting since the additional instances are just copies and don’t provide new information.")

    # Random Under-Sampling
    st.subheader("Random Under-Sampling")
    st.image("/Data/down_sampling.png", caption="Random Under-Sampling")
    st.write("This method reduces the number of instances in the majority class by removing them randomly. It balances the class distribution but can lead to the loss of potentially valuable data, as it indiscriminately discards instances.")

    # SMOTE
    st.subheader("SMOTE (Synthetic Minority Over-sampling Technique)")
    st.image("/Data/smote.png", caption="SMOTE")
    st.write("SMOTE creates new, synthetic instances by interpolating between existing instances in the minority class and their nearest neighbors. This enhances the diversity of the minority class with new, unique examples, thereby decreasing the risk of overfitting.")

    # Conditional Generative Adversarial Networks (cGANs)
    st.subheader("Conditional Generative Adversarial Networks (cGANs)")
    st.image("/Data/cgan.png", caption="cGANs")
    st.write("cGANs use two models—a generator and a discriminator—that work against each other to produce new, synthetic instances of the minority class. The generator creates new data points while the discriminator evaluates their authenticity. This method not only generates diverse and realistic samples but also helps in significantly enhancing the training dataset without repeating existing instances.")


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
