import streamlit as st

def upsampling_description():
    st.title("Upsampling Techniques")
    st.markdown("""
    In our nurse attrition project, it was important to address the imbalance in our dataset. Nurses leaving their jobs happens far less often than them staying, which can lead models to miss these important events. To overcome this, we use advanced upsampling techniques to help our models accurately reflect the realities of job attrition, avoiding bias towards the more common outcome.

    **Enhancing Predictive Accuracy with Upsampling**
    Traditional models typically struggle with rare events, usually predicting the most common scenario. We use upsampling to balance the dataset, improving our models’ ability to learn from the rarer cases of nurses quitting. This method boosts predictive accuracy, helping our models identify key patterns of attrition. We use the following upsampling techniques:""")

    # Random Over-Sampling
    st.subheader("Random Over-Sampling")
    st.image("Data/random_upsampling.png", caption="Random Over-Sampling")
    st.write("This technique duplicates instances from the minority class, such as attrition events, to even out the class distribution. While straightforward, it risks overfitting since the additional instances are just copies and don’t provide new information.")

    # Random Under-Sampling
    st.subheader("Random Under-Sampling")
    st.image("Data/down_sampling.png", caption="Random Under-Sampling")
    st.write("This method reduces the number of instances in the majority class by removing them randomly. It balances the class distribution but can lead to the loss of potentially valuable data, as it indiscriminately discards instances.")

    # SMOTE
    st.subheader("SMOTE (Synthetic Minority Over-sampling Technique)")
    st.image("Data/smote.png", caption="SMOTE")
    st.write("SMOTE creates new, synthetic instances by interpolating between existing instances in the minority class and their nearest neighbors. This enhances the diversity of the minority class with new, unique examples, thereby decreasing the risk of overfitting.")

    # Conditional Generative Adversarial Networks (cGANs)
    st.subheader("Conditional Generative Adversarial Networks (cGANs)")
    st.image("Data/cgan.png", caption="cGANs")
    st.write("cGANs use two models—a generator and a discriminator—that work against each other to produce new, synthetic instances of the minority class. The generator creates new data points while the discriminator evaluates their authenticity. This method not only generates diverse and realistic samples but also helps in significantly enhancing the training dataset without repeating existing instances.")

import streamlit as st
import streamlit as st

def model_description():
    st.title("Model Descriptions")
    st.markdown("""
    Learn how different predictive models work, each designed for its strengths in analyzing various aspects of nurse attrition data:
    """)

    # Logistic Regression
    st.subheader("Logistic Regression")
    st.image("Data/log_reg_xkcd.png")
    st.write("Think of Logistic Regression as weighing scales. It considers different job factors (like workload, satisfaction) and tips towards predicting if a nurse will stay or leave. It’s great for yes/no outcomes, making it ideal for deciding if a nurse might quit.")

    # Support Vector Machine (SVM)
    st.subheader("Support Vector Machine (SVM)")
    st.image("Data/svm_xkcd.png")
    st.write("SVM finds the best boundary that separates nurses who stay from those who leave. Imagine drawing the widest possible road between two groups of people; SVM works similarly to ensure clear separation in complex situations.")

    # K-Nearest Neighbors (KNN)
    st.subheader("K-Nearest Neighbors (KNN)")
    st.image("Data/knn_xkcd.png")
    st.write("KNN looks at a nurse and their nearest co-workers to predict behavior. If many close colleagues have quit, KNN assumes this nurse might too. It’s like predicting someone’s next step by looking at their friends’ choices.")

    # Random Forest
    st.subheader("Random Forest")
    st.image("Data/random_forest_xkcd.png")
    st.write("Random Forest uses a team of decision trees that each look at the data differently. By combining their decisions, it makes a more accurate and robust prediction than any single tree could, much like a wise council making a joint decision.")

    # XGBoost
    st.subheader("XGBoost")
    st.image("Data/xgboost_xkcd.png") #, caption="XGBoost")
    st.write("XGBoost works like a team refining ideas through quick, iterative updates. Each model learns from the mistakes of the one before, gradually improving predictions. It's fast and accurate, making it a favorite in many data challenges.")
