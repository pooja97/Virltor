#uing pretrained model vgg16, pca - for feature extraction and Kmeans

from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input 
import warnings
warnings.filterwarnings("ignore")

# Models
from keras.applications.vgg16 import VGG16 
from keras.models import Model 

# clustering and dimension reduction
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA 
import matplotlib
import os 
import numpy as np 
import matplotlib.pyplot as plt 
import plotly as py
import plotly.graph_objs as go
from sklearn import metrics 
from sklearn.preprocessing import StandardScaler

path = r"/Users/sheshmani/Desktop/virltor/clustering_analysis/segmentation_images"

os.chdir(path)

house_images = []

with os.scandir(path) as files:
    for file in files:
        if file.name.endswith('png'):
            house_images.append(file.name)

# (224,224) because vgg16 model requires our image to be of 224,224 np array
model = VGG16(include_top=False)
model = Model(inputs=model.inputs,outputs = model.layers[-2].output)


# Function for extracting features from the images

def feature_extractor(image_file,model):
    img = load_img(image_file,target_size=(224,224)) 
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    pre_processed_image = preprocess_input(x)

    #extracting the output layer 
    features = model.predict(pre_processed_image,use_multiprocessing=True)
    return features

# Extracting features from each image file and storing it 
# in a dictionary with file_name as the key and the features as a value

data_dictionary = {}
feature_path = r"/Users/sheshmani/Desktop/virltor/clustering_analysis/features"

for houses in house_images:
    feature = feature_extractor(houses,model)
    print(feature.shape)
    data_dictionary[houses] = feature 

filenames = np.array(list(data_dictionary.keys()))
features = np.array(list(data_dictionary.values()))

features = features.reshape(-1,4096)
#feature scaling
scalar = StandardScaler()
features = scalar.fit_transform(features)

# Using PCA for dimensionality reduction 

variance = 0.98
pca = PCA(n_components=5)
pca.fit(features)

transformed_features = pca.transform(features)
print("Dimension of our dataset after PCA: ",str(transformed_features.shape))


k_means = KMeans(init = "k-means++", n_clusters = 40,random_state=22)
k_means.fit(transformed_features)


#saving cluster images in a folder
# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames,k_means.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)


def view_cluster(cluster):
    plt.figure(figsize = (12,8));
    cluster_folder_path = '/Users/sheshmani/Desktop/virltor/clustering_analysis/Cluster_output'
    folder_path = os.path.join(cluster_folder_path,str(cluster))
    if os.path.exists(folder_path):
        pass
    else:
        os.mkdir(folder_path)

    # gets the list of filenames for a cluster
    files = groups[cluster]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(333);
        file_name = os.path.basename(file) 
        img = load_img(file)
        img = np.array(img) 
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(os.path.join(folder_path,file_name))
        matplotlib.pyplot.close()
        # plt.show()
        plt.axis('off')
    matplotlib.pyplot.close()

# for i in range(len(groups)):
#     view_cluster(i)



#  ELBOW METHOD FOR FINDING THE OPTIMAL K (CLUSTER VALUE)
# sse = []
# list_k = list(range(3,50))

# for k in list_k:
#     km = KMeans(init = 'k-means++',n_clusters=k,random_state=22)
#     km.fit(transformed_features)
#     sse.append(km.inertia_)

# # Plot sse against k
# # plt.figure(figsize=(6, 6))
# plt.plot(list_k, sse,'bx-')
# plt.xlabel('Number of clusters')
# plt.ylabel('Sum of squared distance')
# plt.title('Elbow Method for optimal K')
# plt.show()
        


#calculating Silhouette score for each cluster
labels = k_means.labels_
silhouette_score_list = metrics.silhouette_samples(transformed_features,labels)
means_lst = []
for label in range(0,10):
    means_lst.append(silhouette_score_list[labels == label].mean())
print(means_lst)


from yellowbrick.cluster import silhouette_visualizer
for i in range(2,10):
    km = KMeans(n_clusters=i, init='k-means++',random_state=42)
    visualizer = silhouette_visualizer(km,transformed_features, colors='yellowbrick')
    visualizer.show()

from yellowbrick.cluster import KElbowVisualizer
#Elbow and silhouette score plot
km = KMeans(random_state=42)
visualizer = KElbowVisualizer(km, k=(2,39))

visualizer.fit(transformed_features)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure



#3D Plotly Visualization of Clusters using go (first 10 clusters)
layout = go.Layout(
    title='<b>Cluster Visualisation</b>',
    yaxis=dict(
        title='<i>Y</i>'
    ),
    xaxis=dict(
        title='<i>X</i>'
    )
)
colors = ['red','green' ,'blue','purple','magenta','yellow','cyan','maroon','teal','black']
trace = [ go.Scatter3d() for _ in range(50)]
for i in range(0,10):
    my_members = (k_means.labels_ == i)
    index = [h for h, g in enumerate(my_members) if g]
    trace[i] = go.Scatter3d(
            x=transformed_features[my_members, 0],# 0 is a component among the 420 components. Feel free to change it
            y=transformed_features[my_members, 1],# 1 is a component among the 420 components. Feel free to change it
            z=transformed_features[my_members, 2],# 2 is a component among the 420 components. Feel free to change it
            mode='markers',
            marker = dict(size = 2,color = colors[i]),
            hovertext=index,
            name='Cluster'+str(i),
   
            )
fig = go.Figure(data=[trace[0],trace[1],trace[2],trace[3],trace[4],trace[5],trace[6],trace[7],trace[8],trace[9]], layout=layout)
    
py.offline.iplot(fig)
