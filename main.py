import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
from geopandas.datasets import get_path
import folium
from sklearn.cluster import DBSCAN

#Cleaning the data
df=pd.read_excel("fatal-police-shootings-data.xls")
print(df.head(5))
list1 = df.columns
print(list1)
#df_1 = df.dropna()
#print(df_1.to_string()) 
#To remove the rows which are empty 
df.dropna(inplace=True)
#To fix the wrong format in the date
df['date'] = pd.to_datetime(df['date'])
#print(df.to_string())
#print(df.isnull().sum())

print(df.describe)
#Data is clean
#Tackling with non-ordinal data

count_1=df['manner_of_death'].value_counts()
print(count_1)
count_2 = df['gender'].value_counts()
print(count_2)
count_3 = df['flee'].value_counts()
print(count_3)
df = pd.get_dummies(df,columns=['manner_of_death','gender','flee'])
print(df.columns)

#create a decision tree to explain the analysis for categorical data






df_cal=df[df['state']=='CA']
# #create a new datframe for geo spatial analysis
# latitude, longitude = df_cal['latitude'].tolist(), df_cal['longitude'].tolist()
# map_center = [36.7783,-119.4179]
# mymap = folium.Map(location=map_center, zoom_start=6)
# for lat, lon in zip(latitude, longitude):
#     folium.CircleMarker(location=[lat, lon], radius=1, color='blue', fill=True, fill_color='blue').add_to(mymap)
# mymap.show_in_browser()




# Assuming your data is in a DataFrame called 'df' with columns 'latitude' and 'longitude'
latitude, longitude = df_cal['latitude'].values, df_cal['longitude'].values

# Combine latitude and longitude into a single array for DBSCAN
coordinates = list(zip(latitude, longitude))
print(coordinates.count)
# Convert the coordinates to radians (required for haversine distance)
from math import radians
coordinates_in_radians = [tuple(map(radians, coord)) for coord in coordinates]

# Apply DBSCAN clustering
epsilon = 1.5  # Adjust as needed
min_samples = 5  # Adjust as needed
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
clusters = dbscan.fit_predict(coordinates_in_radians)

# Add the cluster labels to the DataFrame
df['cluster'] = clusters

# Create a map centered on the average coordinates
map_center = [sum(latitude) / len(latitude), sum(longitude) / len(longitude)]
mymap = folium.Map(location=map_center, zoom_start=10)

# Add markers for each point with different colors based on the cluster
for lat, lon, cluster in zip(latitude, longitude, clusters):
    color = 'red' if cluster == -1 else f'cluster_{cluster}'
    folium.CircleMarker(location=[lat, lon], radius=5, color=color, fill=True, fill_color=color).add_to(mymap)

# Save the map to an HTML file or display it
mymap.save('map_with_dbscan_clusters.html')
# OR
mymap