import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans

mnist = tf.keras.datasets.mnist

# From this website but modified
# https://medium.datadriveninvestor.com/k-means-clustering-for-imagery-analysis-56c9976f16b6
 
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Conversion to float and normalization 
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')

X = x_train.reshape(len(x_train),-1)
Y = y_train

test_images = x_test.reshape(len(x_test), -1)
test_label = y_test

# normalize the data to 0 - 1

X = X / 255.0
test_images = test_images / 255.0


###########################################################
# set number of clusters 
# (increase n_clusters -> increase accuracy)

n_clusters = 512
# batch_size = > 256 * number of cores of CPU (6)

kmeans = MiniBatchKMeans(n_clusters = n_clusters, init = 'k-means++', batch_size = 3072, random_state = 42, 
                         verbose = 2).fit(X)

# MiniBatchKMeans is faster than kmeans
# kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = 42).fit(X)

centroids = kmeans.cluster_centers_
centroids = centroids.reshape(n_clusters,28,28)
centroids = centroids * 255
centroids = centroids.astype(np.uint8)


def infer_cluster_labels(kmeans, actual_labels):
    inferred_labels = {}

    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]
    
    return inferred_labels

def infer_data_labels(X_labels, cluster_labels):
  # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
            
    return predicted_labels


cluster_labels = infer_cluster_labels(kmeans, Y)
X_clusters = kmeans.predict(test_images)
predict_labels = infer_data_labels(X_clusters, cluster_labels)

index = 200
row = 8
col = 8

plt.figure (figsize = (15,15))
for i in range(0, row * col):
    plt.subplot(row, col, i + 1)
    plt.title('Pre:' + str(predict_labels[index + i]) + 
              '/ Act: ' + str(y_test[index + i]))
    plt.imshow(test_images[index+ i].reshape(28,28,1), cmap='gray')
    plt.grid(False)
    plt.axis('off')

plt.show()



