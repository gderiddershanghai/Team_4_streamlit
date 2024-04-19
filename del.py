import streamlit as st
from ml_logic.preprocessor import preprocess_and_split
from ml_logic.upsampling import upsample_data
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
# Add imports for each model class
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def show_results():
    if 'probabilities' in st.session_state:
        threshold = st.slider('Set Decision Threshold', 0.0, 1.0, 0.5, 0.01)
        predictions = (st.session_state['probabilities'] >= threshold).astype(int)
    else:
        predictions = st.session_state['predictions']

    accuracy = accuracy_score(st.session_state['y_test'], predictions)
    precision = precision_score(st.session_state['y_test'], predictions, zero_division=0)
    recall = recall_score(st.session_state['y_test'], predictions, zero_division=0)
    f1 = f1_score(st.session_state['y_test'], predictions, zero_division=0)

    st.write("Metrics:")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}")

    # Display confusion matrix
    cm = confusion_matrix(st.session_state['y_test'], predictions)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    ax.yaxis.set_ticklabels(['Negative', 'Positive'])
    st.pyplot(fig)

# Define the list of variables
variables = ['Age', 'BusinessTravel', 'DailyRate',
             'Department', 'DistanceFromHome', 'Education', 'EducationField',
             'EmployeeCount', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
             'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
             'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
             'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
             'RelationshipSatisfaction', 'StandardHours', 'Shift',
             'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
             'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
             'YearsWithCurrManager']

model_classes = {
    'Logistic Regression': LogisticRegression,
    'Random Forest': RandomForestClassifier,
    'KNN Classifier': KNeighborsClassifier,
    'SVM Classifier': SVC,
    'XGBoost': XGBClassifier
}


# Custom CSS to inject into the Streamlit app for styling
st.markdown(
    """
    <style>
    .column {
        color: blue;
    }
    /* Add other CSS styles as needed */
    </style>
    """,
    unsafe_allow_html=True
)

# Function to create checkboxes within columns, with preselection for specified variables
def create_checkboxes(variables, columns=3, preselected=None):
    if preselected is None:
        preselected = []
    selected_variables = []
    cols = st.columns(columns)
    for index, variable in enumerate(variables):
        col = cols[index % columns]
        # Preselect specified variables by setting the value parameter to True if the variable is in the preselected list
        is_selected = col.checkbox(variable, key=variable, value=variable in preselected)
        if is_selected:
            selected_variables.append(variable)
    return selected_variables


st.title("Select Variables")
preselected_variables = [
     'Age', 'DistanceFromHome', 'MonthlyIncome', 'OverTime', 'Shift', 'EnvironmentSatisfaction','TotalWorkingYears']

selected_variables = create_checkboxes(variables, preselected=preselected_variables)

if st.button('Submit Variables'):
    st.session_state['selected_variables'] = ['Attrition'] + selected_variables
    fp = 'Data/watson_healthcare_modified.csv'
    X_train, X_test, y_train, y_test = preprocess_and_split(fp=fp, columns_to_use=st.session_state['selected_variables'])
    st.session_state['X_train'], st.session_state['X_test'] = X_train, X_test
    st.session_state['y_train'], st.session_state['y_test'] = y_train, y_test
    st.success("Data processed successfully!")

if 'selected_variables' in st.session_state:
    st.subheader("Select Upsampling Technique")
    upsampling_techniques = {
        'None': "No upsampling applied",
        "Random Upsampling": "Increases the number of instances in the minority class by randomly replicating them.",
        "Random Downsampling": "Reduces the number of instances in the majority class by randomly removing them.",
        "SMOTE": "Synthetic Minority Over-sampling Technique. Generates synthetic examples rather than over-sampling with replacement.",
        "CGAN": "Conditional Generative Adversarial Networks. Uses a GAN framework to generate synthetic data conditioned on the class."
    }
    selected_technique = st.selectbox("Choose an upsampling technique:", list(upsampling_techniques.keys()))
    st.write("Description:", upsampling_techniques[selected_technique])

    if st.button('Apply Upsampling'):
        methods = {'None': 'none', "Random Upsampling": 'undersample', "Random Downsampling": 'oversample', "SMOTE": 'smote', "CGAN": 'cgan'}
        X_train_resampled, y_train_resampled = upsample_data(st.session_state['X_train'], st.session_state['y_train'], method=methods[selected_technique])
        st.session_state['X_train_resampled'], st.session_state['y_train_resampled'] = X_train_resampled, y_train_resampled
        st.success(f"Upsampling {selected_technique} applied successfully!")


# Add a visual separator
st.markdown("---")


# Define your model selection UI somewhere above this code
model = st.radio("Select the model you want to use:",
                 ['Logistic Regression', 'Random Forest', 'KNN Classifier', 'SVM Classifier', 'XGBoost'])

if 'X_train_resampled' in st.session_state:
    # Submit button for model training
    if st.button('Train Model', key=f'train_{model}'):
        # Initialize the selected model
        if model == 'Logistic Regression':
            trained_model = LogisticRegression()
        elif model == 'Random Forest':
            trained_model = RandomForestClassifier()
        elif model == 'KNN Classifier':
            trained_model = KNeighborsClassifier()
        elif model == 'SVM Classifier':
            trained_model = SVC(probability=True)  # Ensure to enable probability for SVC
        elif model == 'XGBoost':
            trained_model = XGBClassifier()

        # Fit the model
        trained_model.fit(st.session_state['X_train_resampled'], st.session_state['y_train_resampled'])

        # Post-training actions
        if model in ['Logistic Regression', 'SVM Classifier']:
            # These models support probability estimates
            probabilities = trained_model.predict_proba(st.session_state['X_test'])[:, 1]
            st.session_state['probabilities'] = probabilities
            st.success("Model trained. Adjust the threshold slider to see changes in the confusion matrix.")
        else:
            # These models do not support probability estimates by default
            predictions = trained_model.predict(st.session_state['X_test'])
            st.session_state['predictions'] = predictions
            st.success("Model trained. View predictions and metrics below.")

# Show additional results or metrics
if 'probabilities' in st.session_state:


    threshold = st.slider('Set Decision Threshold', 0.0, 1.0, 0.5, 0.05)
    predictions = (st.session_state['probabilities'] >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(st.session_state['y_test'], predictions)
    precision = precision_score(st.session_state['y_test'], predictions, zero_division=0)
    recall = recall_score(st.session_state['y_test'], predictions, zero_division=0)
    f1 = f1_score(st.session_state['y_test'], predictions, zero_division=0)

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}")

    # Confusion matrix visualization
    cm = confusion_matrix(st.session_state['y_test'], predictions)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    ax.yaxis.set_ticklabels(['Negative', 'Positive'])
    st.pyplot(fig)


    thresholds = np.linspace(0, 1, 50)

    # Lists to store the metrics for each threshold
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Calculate metrics for each threshold
    for threshold in thresholds:
        predictions = (st.session_state['probabilities'] >= threshold).astype(int)
        accuracies.append(accuracy_score(st.session_state['y_test'], predictions))
        precisions.append(precision_score(st.session_state['y_test'], predictions, zero_division=0))
        recalls.append(recall_score(st.session_state['y_test'], predictions, zero_division=0))
        f1_scores.append(f1_score(st.session_state['y_test'], predictions, zero_division=0))

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.title('Metrics as a function of the decision threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)

    # Show the plot in Streamlit
    st.pyplot(plt)


    # threshold = st.slider('Set Decision Threshold', 0.0, 1.0, 0.5, 0.05)
    # predictions = (st.session_state['probabilities'] >= threshold).astype(int)

    # # Calculate metrics
    # accuracy = accuracy_score(st.session_state['y_test'], predictions)
    # precision = precision_score(st.session_state['y_test'], predictions, zero_division=0)
    # recall = recall_score(st.session_state['y_test'], predictions, zero_division=0)
    # f1 = f1_score(st.session_state['y_test'], predictions, zero_division=0)

    # # Display metrics in columns
    # col1, col2, col3, col4 = st.columns(4)
    # col1.metric("Accuracy", f"{accuracy:.2%}")
    # col2.metric("Precision", f"{precision:.2%}")
    # col3.metric("Recall", f"{recall:.2%}")
    # col4.metric("F1 Score", f"{f1:.2%}")

    # # Confusion matrix visualization
    # cm = confusion_matrix(st.session_state['y_test'], predictions)
    # fig, ax = plt.subplots()
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    # ax.set_title('Confusion Matrix')
    # ax.set_xlabel('Predicted Labels')
    # ax.set_ylabel('True Labels')
    # ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    # ax.yaxis.set_ticklabels(['Negative', 'Positive'])
    # st.pyplot(fig)
