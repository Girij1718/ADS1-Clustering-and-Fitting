import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\girij\Downloads\Clustering\Adidas Vs Nike.csv"
data = pd.read_csv(file_path)

# Inspect the first few rows of the dataset
data.head()

# Check for missing values
data.isnull().sum()

# Drop rows with missing values (or impute if necessary)
data = data.dropna()  # Alternatively, use data.fillna() for imputation

# Normalize the numerical columns
scaler = MinMaxScaler()
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

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

# Strip any extra spaces from column names
data.columns = data.columns.str.strip()

# Identify the numerical columns (for clustering, only numerical columns are relevant)
numerical_columns = data.select_dtypes(include=['number', 'float64', 'int64']).columns

# Call the function to plot the Elbow Curve
plot_elbow_curve(data, numerical_columns, max_clusters=10)

# Let's assume the optimal number of clusters is 3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(data[numerical_columns])

# Compute the silhouette score
silhouette_avg = silhouette_score(data[numerical_columns], clusters)
print(f"Silhouette Score for {optimal_k} clusters: {silhouette_avg}")

def linear_regression_plot(data, x_column, y_column):
    """
    Fits a linear regression model and plots the scatter plot with regression line.
    Parameters:
    data (DataFrame): The input DataFrame containing the data.
    x_column (str): The name of the independent variable (X-axis).
    y_column (str): The name of the dependent variable (Y-axis).
    """
    # Extract the independent (X) and dependent (Y) variables
    x = data[x_column].values.reshape(-1, 1)
    y = data[y_column].values

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(x, y)

    # Make predictions using the model
    y_pred = model.predict(x)

    # Plot the scatter plot and regression line
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, y_pred, color='red', label='Regression line')
    plt.title(f'Linear Regression: {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.show()

    # Print the slope and intercept of the regression line
    print(f"Slope (m): {model.coef_[0]}")
    print(f"Intercept (c): {model.intercept_}")

# Strip any extra spaces from column names
data.columns = data.columns.str.strip()

# Print the cleaned column names to check
print("Column Names:", data.columns)

linear_regression_plot(data, 'Listing Price', 'Sale Price')

def plot_histogram(data, column_name, bins=20, color='blue', alpha=0.7):
    """
    Function to plot a histogram for a specified column in the DataFrame.
    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column to plot.
    - bins (int): Number of bins for the histogram.
    - color (str): Color of the histogram bars.
    - alpha (float): Transparency of the bars (0 to 1).
    """
    plt.hist(data[column_name], bins=bins, color=color, alpha=alpha)
    plt.title(f"Distribution of {column_name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

plot_histogram(data, 'Sale Price', bins=20, color='blue', alpha=0.7)

def plot_correlation_heatmap(data):
    """
    Plots a correlation heatmap for numeric columns in the given DataFrame.
    Parameters:
    data (DataFrame): The input DataFrame containing the data.
    """
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=['number', 'float64', 'int64'])

    # Compute correlation matrix for numeric columns only
    corr_matrix = numeric_data.corr()

    # Plot heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        annot_kws={"size": 10, "color": "black"}, 
        cmap="coolwarm",
        center=0, 
        linewidths=0.5, 
        linecolor='black', 
        cbar_kws={'shrink': 0.8}
    )
    plt.title("Correlation Heatmap", fontsize=16)
    plt.yticks(rotation=0)
    plt.show()

plot_correlation_heatmap(data)

def create_pie_chart(file_path, column_name):
    """
    Function to load a CSV file and create a pie chart for a specific column.
    Parameters:
    - file_path (str): The path to the CSV file.
    - column_name (str): The name of the column to visualize.
    """
    # Step 1: Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    
    # Step 2: Count the unique values in the specified column
    value_counts = data[column_name].value_counts()
    
    # Step 3: Create a pie chart
    plt.figure(figsize=(8, 8))  # Set the size of the pie chart
    plt.pie(
        value_counts, 
        labels=value_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=plt.cm.Pastel1.colors  # Use a nice pastel color palette
    )
    plt.title(f'Distribution of {column_name}', fontsize=16)
    plt.show()

create_pie_chart(file_path, 'Brand')
