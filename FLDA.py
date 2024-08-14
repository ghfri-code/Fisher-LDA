from PIL import Image
from matplotlib import cm
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import matplotlib.pyplot as plt

#------------------ loading the whole image dataset ---------------------------#
def load_images(path=None):
    if path is None:
        path = 'Data'
    else:
        pass
    # for storing images
    imgs = []
    # for storing labels
    labels = []
    # Loads and convert to gray at the same time
    for img_name in os.listdir(path):
        if "AN" in img_name:
            labels.append(0)        
            imgs.append(Image.open(str(path) + '/' + img_name).convert("L").resize((64,64)))
        elif "DI" in img_name:
            labels.append(1)        
            imgs.append(Image.open(str(path) + '/' + img_name).convert("L").resize((64,64)))
        elif "FE" in img_name:
            labels.append(2)        
            imgs.append(Image.open(str(path) + '/' + img_name).convert("L").resize((64,64)))
        elif "HA" in img_name:
            labels.append(3)        
            imgs.append(Image.open(str(path) + '/' + img_name).convert("L").resize((64,64)))
        elif "NE" in img_name:
            labels.append(4)        
            imgs.append(Image.open(str(path) + '/' + img_name).convert("L").resize((64,64)))
        elif "SA" in img_name:
            labels.append(5)        
            imgs.append(Image.open(str(path) + '/' + img_name).convert("L").resize((64,64)))
        elif "SU" in img_name:
            labels.append(6)        
            imgs.append(Image.open(str(path) + '/' + img_name).convert("L").resize((64,64)))
            
    return labels, imgs

labels, imgs = load_images('.\jaffe')
uniqueClass = np.unique(labels)
labels = np.array(labels)
#--------------------- vectorization -----------------------------------#
images = []

def vectorization(imgs):

    for image in imgs:
        img = np.array(image).flatten() 
        images.append(img)
        
    return images
    
    
faces_matrix = np.vstack(vectorization(imgs))
mean_face = np.mean(faces_matrix, axis=0)
feature_no = faces_matrix.shape[1]

#----------------------- Mean vector for each class -------------------#
np.set_printoptions(precision=4)
mean_vectors = []
def mean_for_each_class(faces):
    
    for cl in range(0,7):
        mean_vectors.append(np.mean(faces[labels==cl], axis=0))
        print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl]))
        print(np.shape(mean_vectors[cl]))
    return mean_vectors
  
mean_vectors = mean_for_each_class(faces_matrix)

#-------------------- WithinScatterMatrices --------------------------#

def computeWithinScatterMatrices(X, y, uniqueClass, mean_vectors):

    s_matrix = []
    for cls, mean in enumerate(mean_vectors):
        Si = np.zeros((X.shape[1], X.shape[1]))
        for row in X[y == cls]:
            t = (row - mean).reshape(1, X.shape[1])
            Si += np.dot(t.T, t)
        s_matrix.append(Si)
        
    S_W = np.zeros((X.shape[1], X.shape[1]))
    for s_i in s_matrix:
        S_W += s_i                  # sum class scatter matrices

    return S_W

S_W = computeWithinScatterMatrices(faces_matrix, labels, uniqueClass, mean_vectors)

#--------------------------- BetweenClassScatterMatrices -----------------------------------#

def computeBetweenClassScatterMatrices(X, y, mean_vectors):
    
    data_mean = np.mean(X, axis=0).reshape(1, X.shape[1])
    S_B = np.zeros((X.shape[1], X.shape[1]))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y == i].shape[0]
        mean_vec = mean_vec.reshape(1, X.shape[1])
        mu1_mu2 = mean_vec - data_mean
     
        S_B += n * np.dot(mu1_mu2.T, mu1_mu2)

    return S_B

S_B = computeBetweenClassScatterMatrices(faces_matrix, labels, mean_vectors)

#------------------------------------ eigen_vecs & values -------------------------------#

fisher = np.dot(np.linalg.inv(S_W), S_B)
u, s, v = np.linalg.svd(fisher)

#----------------------------- reconstruct -------------------------------#

def reconstruct(faces_matrix, k):
    # images reconstructed
    reconstructed = np.matrix(u[:, :k]) * np.diag(s[:k]) * np.matrix(v[:k, :])
    return reconstructed
    
#---------------------------- plot reconstructed photo ------------------------

def plot_reconstructed_photos(faces):
    
    for index in range(5):
        fig, axs = plt.subplots(1,3,figsize=(15,6))
        for k, i in zip([1,6,29],np.arange(3)):
            weight = faces[index,:].dot(v[:,:k]) # Get PC scores of the images
            projected_face = weight.dot(v[:,:k].T) # Reconstruct first face in dataset using k PCs
            reconstructed_images = projected_face.reshape(64,64) + mean_face.reshape(64,64)
            ax = plt.subplot(1,3,i+1)
            ax.set_title("k = "+str(k))
            plt.imshow(projected_face.reshape(64,64)+ mean_face.reshape(64,64),cmap='gray');

        fig.suptitle(("Reconstruction with Increasing Eigenfaces"), fontsize=16);
        plt.show()
    return reconstructed_images

plot_reconstructed_photos(faces_matrix)

#------------------------ mse -----------------------------#

# calculates MSE for one sample
def mse(faces, rec):
    return np.power(np.subtract(faces, rec), 2).mean(axis=None)


# plots the MSE in terms of eigen vectors
def mse_analysis(faces):
    x = []
    y = []
    idx = np.linspace(1, 530, 200)
    for i in idx:
        rec = reconstruct(faces, k=int(i))
        x.append(int(i))
        y.append(mse(faces, rec))

    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(14, 9))
    pyplot.ylabel('MSE')
    plt.xlabel('K = Number of eigenvectors (Reduced dim.)')
    plt.plot(x, y, label='MSE in terms of # of eigenvectors', alpha=0.8)
    plt.legend(loc='best')
    #plt.savefig('pca_mse')
    plt.show()
    
mse_analysis(fisher)
