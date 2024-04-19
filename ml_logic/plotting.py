import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_PCA(X_train, y_train):
    """
    Plots a 2D PCA visualization of the training data, highlighting the distinction
    between employees who quit and those who stayed.

    This function applies Principal Component Analysis (PCA) to reduce the dimensionality
    of the training data to two dimensions for visualization purposes. It then plots the
    transformed features, coloring the points based on the employee's status (quit or stayed).

    Parameters:
    - X_train (array-like): Training input samples, where n_samples is the number of samples
      and n_features is the number of features.
    - y_train (array-like): Target values (1 for employees who quit, 0 for those who stayed).
      This array should have shape (n_samples,).

    The function does not return any value but displays a matplotlib plot.
    """

    # Define the number of components for PCA and initialize the PCA transformer
    n_pca_components = 2
    pca = PCA(n_components=n_pca_components)

    # Fit the PCA model to the data and apply the dimensionality reduction on X_train
    pca.fit(X_train)
    feature_data = pca.transform(X_train)

    # Split the PCA-transformed data into two groups based on the target variable (y_train)
    feature_data_quit = feature_data[y_train == 1]  # Data for employees who quit
    feature_data_stayed = feature_data[y_train == 0]  # Data for employees who stayed

    # Plot the two groups with different colors and labels to indicate quitting status
    plt.scatter(feature_data_quit[:, 0], feature_data_quit[:, 1], c='red', alpha=0.5, label='Quit')
    plt.scatter(feature_data_stayed[:, 0], feature_data_stayed[:, 1], c='blue', alpha=0.5, label='Stayed')

    # Set the labels for the axes, the plot title, and display the legend
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Training Data')
    plt.legend(loc='best')  # Optimal legend positioning

    # Display the plot
    plt.show()
