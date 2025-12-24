import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")

print("veri seti : ")
print(df.head())

print("\nVeri bilgisi :")
print(df.info())

print("\nistatiksel özet :")
print(df.describe().round(2))

ozet =df.describe().round(2)

df.drop("CustomerID",axis=1,inplace=True)

X = df[["Annual Income (k$)","Spending Score (1-100)"]].values

inertia_list = []

for k in range(1,11):#1 dahil 11 haric
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(X)
    inertia_list.append(kmeans.inertia_)
    
plt.figure(figsize=(10,6))
plt.plot(range(1,11),inertia_list,marker='o',color='pink')  
plt.title('grup sayısı karar grafiği')
plt.xlabel('küme sayısı')
plt.ylabel('hata puanı')


kmeans=KMeans(n_clusters=5,random_state=42)
kmeans.fit(X)
y_kmeans=kmeans.predict(X)

#model goresellestirme
plt.figure(figsize=(12, 8))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Orta Gelir, Orta Harcama')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Yüksek Gelir, Yüksek Harcama')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Düşük Gelir, Yüksek Harcama')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='purple', label='Yüksek Gelir, Düşük Harcama')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='yellow', label='Düşük Gelir, Düşük Harcama')
plt.title('Müşteri Grupları (Segmentasyon)')
plt.xlabel('Yıllık Gelir (k$)')
plt.ylabel('Harcama Skoru (1-100)')
plt.grid(True, alpha=0.3)#izgara eklemek icin 
plt.legend()
plt.show()


