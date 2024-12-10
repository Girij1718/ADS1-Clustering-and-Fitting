import pandas as pd

# Load the dataset
file_path = r"C:\Users\girij\Downloads\Clustering\Adidas Vs Nike.csv"
data = pd.read_csv(file_path)

# Inspect the first few rows of the dataset
data.head()

# Check for missing values
data.isnull().sum()

# Drop rows with missing values (or impute if necessary)
data = data.dropna()  # Alternatively, use data.fillna() for imputation

from sklearn.preprocessing import MinMaxScaler

# Normalize the numerical columns
scaler = MinMaxScaler()
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

def plot_elbow_curve(data, numerical_columns, max_clusters=10):
    """
    Plots the Elbow Curve to determine the optimal number of clusters for KMeans.
    
    Parameters:
    data (DataFrame): The input DataFrame containing the data.
    numerical_columns (list): List of column names to be used for clustering.
    max_clusters (int): The maximum number of clusters to try (default is 10).
    """
    distortions = []  # To store inertia values for each k
    for k in range(1, max_clusters + 1):  # Try from 1 to max_clusters clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data[numerical_columns])  # Fit only the specified numerical columns
        distortions.append(kmeans.inertia_)  # Store the inertia value
    
    # Plot the Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.title("Elbow Method to Find Optimal Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Distortion (Inertia)")
    plt.show()


# Load the dataset
file_path = r"C:\Users\girij\Downloads\Clustering\Adidas Vs Nike.csv"
data = pd.read_csv(file_path)

# Strip any extra spaces from column names
data.columns = data.columns.str.strip()

# Identify the numerical columns (for clustering, only numerical columns are relevant)
numerical_columns = data.select_dtypes(include=['number', 'float64', 'int64']).columns

# Call the function to plot the Elbow Curve
plot_elbow_curve(data, numerical_columns, max_clusters=10)



