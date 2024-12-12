import pandas as pd  # for data manipulation
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for heatmaps (optional)
from sklearn.linear_model import LinearRegression  # for linear regression
from sklearn.preprocessing import StandardScaler  # for data scaling
from sklearn.cluster import KMeans  # for K-means clustering
from sklearn.metrics import silhouette_score  # for silhouette score calculation
from sklearn.decomposition import PCA  # for PCA


df = pd.read_csv('world_bank_dataset.csv')

def data_overview_plots():
    """
    Create visualizations to provide an overview of the dataset
    """
    plt.figure(figsize=(15, 10))
    
    # 1. Histogram of GDP Distribution
    plt.subplot(2, 2, 1)
    df['GDP (USD)'].hist(bins=20, edgecolor='black')
    plt.title('GDP Distribution')
    plt.xlabel('GDP (USD)')
    plt.ylabel('Frequency')
    
    # 2. Pie Chart of Countries Representation
    plt.subplot(2, 2, 2)
    country_counts = df['Country'].value_counts()
    plt.pie(country_counts.values[:10], labels=country_counts.index[:10], autopct='%1.1f%%')
    plt.title('Top 10 Countries Representation')
    
    # 3. Bar Plot of Average Life Expectancy by Country
    plt.subplot(2, 2, 3)
    avg_life_expectancy = df.groupby('Country')['Life Expectancy'].mean().sort_values(ascending=False)
    avg_life_expectancy[:10].plot(kind='bar')
    plt.title('Average Life Expectancy by Country')
    plt.xlabel('Country')
    plt.ylabel('Life Expectancy')
    plt.xticks(rotation=45, ha='right')
    
    # 4. Box Plot of CO2 Emissions
    plt.subplot(2, 2, 4)
    df.boxplot(column='CO2 Emissions (metric tons per capita)')
    plt.title('CO2 Emissions Distribution')
    
    plt.tight_layout()
    plt.savefig('data_overview.png')
    plt.show()

def correlation_and_regression():
    """
    Create correlation heatmap and perform linear regression
    """
    # Select numeric columns for correlation
    numeric_cols = ['GDP (USD)', 'Population', 'Life Expectancy', 
                    'Unemployment Rate (%)', 'CO2 Emissions (metric tons per capita)', 
                    'Access to Electricity (%)']
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Economic Indicators')
    plt.tight_layout()
    plt.savefig('heatmap.png')
    plt.show()
    
    # Linear Regression: GDP vs Life Expectancy
    X = df['GDP (USD)'].values.reshape(-1, 1)
    y = df['Life Expectancy'].values
    
    # Fit linear regression
    reg = LinearRegression().fit(X, y)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.5)
    plt.plot(X, reg.predict(X), color='red', linewidth=2)
    plt.title('GDP vs Life Expectancy')
    plt.xlabel('GDP (USD)')
    plt.ylabel('Life Expectancy')
    plt.savefig('linear_regression.png')
    plt.show()
    
    print(f"Linear Regression Results:")
    print(f"Slope (Coefficient): {reg.coef_[0]}")
    print(f"Intercept: {reg.intercept_}")

def kmeans_clustering():
    """
    Perform K-means clustering and visualize results
    """
    # Prepare data for clustering
    features = ['GDP (USD)', 'Population', 'Life Expectancy', 
                'Unemployment Rate (%)', 'CO2 Emissions (metric tons per capita)']
    
    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # Elbow method to find optimal number of clusters
    inertias = []
    silhouette_scores = []
    k_range = range(2, 10)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
    
    # Elbow Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    # Silhouette Score Plot
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'rx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    
    plt.tight_layout()
    plt.savefig('elbow_silhouette.png')
    plt.show()
    
    # Perform clustering with optimal k (let's choose 4 based on plots)
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # Visualize clusters using PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df['Cluster'], cmap='viridis')
    plt.title('Country Clusters based on Economic Indicators')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    plt.savefig('clusters_using.png')
    plt.show()

# Run the analysis functions
data_overview_plots()
correlation_and_regression()
kmeans_clustering()