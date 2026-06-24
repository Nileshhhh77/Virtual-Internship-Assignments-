# Name: Nilesh Desai
# Virtual Internship Assignment - Week 3
# Topic: K-Means Clustering and PCA
# Dataset: Iris Dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('iris.csv')

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\n========================================")
print("   TASK 1: K-MEANS CLUSTERING")
print("========================================")

# We use only the numerical feature columns for clustering
# K-Means cannot work with text/label columns
feature_cols = [col for col in df.columns if col not in ['Species', 'species', 'class', 'Id', 'id']]
print(f"\nFeatures used for clustering: {feature_cols}")

X = df[feature_cols]

# Applying K-Means with k=3 because Iris has 3 species
# random_state=42 ensures same result every time we run
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

print("\nCluster assigned to each row (first 10):")
print(df['Cluster'].head(10))

print("\nNumber of samples in each cluster:")
print(df['Cluster'].value_counts())


print("\n========================================")
print("   TASK 2: PCA - DIMENSION REDUCTION")
print("========================================")

# PCA reduces the 4 feature columns down to 2
# so we can plot them on a 2D scatter plot easily
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"\nOriginal number of features: {X.shape[1]}")
print(f"Reduced number of features after PCA: 2")
print(f"\nVariance explained by each component: {pca.explained_variance_ratio_}")
print(f"Total variance retained: {sum(pca.explained_variance_ratio_):.4f}")

# Add PCA components to dataframe
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

print("\n========================================")
print("   MINI PROJECT 3: IRIS CLUSTERING PROJECT")
print("========================================")

# --- Plot 1: K-Means Clusters using PCA components ---
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for cluster in range(3):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'],
                c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clusters on Iris Dataset (PCA Reduced)')
plt.legend()
plt.tight_layout()
plt.savefig('kmeans_clusters.png')
print("\n K-Means cluster plot saved as 'kmeans_clusters.png'")

# --- Plot 2: True Labels vs Predicted Clusters ---
# Encode the actual species column into numbers for plotting
species_col = None
for col in ['Species', 'species', 'class']:
    if col in df.columns:
        species_col = col
        break

if species_col:
    le = LabelEncoder()
    df['True_Label'] = le.fit_transform(df[species_col])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual species
    for label in range(3):
        label_data = df[df['True_Label'] == label]
        axes[0].scatter(label_data['PCA1'], label_data['PCA2'],
                        label=le.classes_[label], alpha=0.6)
    axes[0].set_title('Actual Species Labels')
    axes[0].set_xlabel('PCA Component 1')
    axes[0].set_ylabel('PCA Component 2')
    axes[0].legend()

    # Predicted clusters
    for cluster in range(3):
        cluster_data = df[df['Cluster'] == cluster]
        axes[1].scatter(cluster_data['PCA1'], cluster_data['PCA2'],
                        c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)
    axes[1].set_title('K-Means Predicted Clusters')
    axes[1].set_xlabel('PCA Component 1')
    axes[1].set_ylabel('PCA Component 2')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('actual_vs_predicted_clusters.png')
    print(" Actual vs Predicted plot saved as 'actual_vs_predicted_clusters.png'")

print("\nSteps performed in this project:")
print("1. Loaded the Iris dataset")
print("2. Selected numerical feature columns for clustering")
print("3. Applied K-Means clustering with k=3")
print("4. Applied PCA to reduce 4 dimensions down to 2")
print("5. Visualized K-Means clusters using scatter plot")
print("6. Compared predicted clusters vs true species labels")

print("\n Week 3 Assignment Completed Successfully!")