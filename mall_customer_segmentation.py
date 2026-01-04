import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

print("Dataset Preview:")
print(df.head())

print("\nData Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe().round(2))

summary = df.describe().round(2)

# Data Cleaning: Remove unnecessary columns
df.drop("CustomerID", axis=1, inplace=True)

# Feature Selection: Annual Income and Spending Score
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

inertia_list = []

# Finding the optimal number of clusters using the Elbow Method
for k in range(1, 11):  # From 1 to 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia_list.append(kmeans.inertia_)
    
# Plotting the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia_list, marker='o', color='pink')  
plt.title('Elbow Method for Optimal Cluster Selection')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Error Score)')

# Training the K-Means model with the optimal cluster count (k=5)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Adding the segment labels to the dataframe
df['Customer_Segment'] = y_kmeans
print("\nUpdated Dataset with Segments:")
print(df.head())

# Model Visualization
plt.figure(figsize=(12, 8))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Average Income, Average Spending')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='High Income, High Spending')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Low Income, High Spending')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='purple', label='High Income, Low Spending')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='yellow', label='Low Income, Low Spending')

plt.title('Customer Groups (Segmentation)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

# Plotting the Centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='black', label='Centroids', marker='.')

plt.grid(True, alpha=0.3) # Adding grid for better readability
plt.legend()
plt.show()
