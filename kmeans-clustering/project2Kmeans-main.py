"""
@author: Ashwin Srinivasan
"""
#importing header files and data preparation
import numpy as np
import scipy.io
import math
from copy import deepcopy
import sys
import warnings
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

data = scipy.io.loadmat('AllSamples.mat')
X = list(list(data.items())[3][1])
tX = []
tY = []
for i in range(0,len(X)):
    tX.append(X[i][0])
    tY.append(X[i][1])
X1 = np.array(list(zip(tX,tY)))
"""
function to compute the objective function to find optimal k value
"""
def objectFunc(data, mean , di):
    s_err = 0
    for i in range(0, len(data)):
          s_err = s_err + errorRate(data[i][0], data[i][1],
                                                         mean[int(di[i])][0],
                                                         mean[int(di[i])][1])
    return s_err
"""
function to calculate the total error
"""
def errorRate(x1, y1, x2 , y2):
    power1 = math.pow((x1-x2), 2)
    power2 = math.pow((y1-y2),2)
    totalPower = power1 + power2
    return totalPower

"""
Function to estimate the Euclidean distance
"""
def euclideanDist(x, y, ax=1):
    return np.linalg.norm(x - y, axis=ax)

"""
Centroid initialization for statergy 1 - Random Initialization
"""
def initialize_random(k):
    Xc = np.random.randint(0, np.max(X1), size=k)
    Xy = np.random.randint(0, np.max(X1), size=k)
    Centroid = np.array(list(zip(Xc, Xy)),dtype= np.int)
    return Centroid

"""
Centroid initialization for statergy 2 - kmeans++
"""
def initialize_optimal(data, k): 
    C = [] 
    C.append(data[np.random.randint( 
            data.shape[0]), :])
    for iterate in range(k - 1): 
        distance = [] 
        for i in range(data.shape[0]): 
            p = data[i, :] 
            d1 = sys.maxsize
            for j in range(len(C)): 
                temp_dist = euclideanDist(p, C[j],None) 
                d1 = min(d1, temp_dist) 
            distance.append(d1) 
        distance = np.array(distance) 
        NC = data[np.argmax(distance), :] 
        C.append(NC) 
        distance = [] 
    return C
"""
Function for kmeans algorithm
Here the calculation is done till two values turns out to be similar that 
is till error value becomes 0 [Till Convergence]
"""
def kmeans(K,stratergy):
    k = K
    if stratergy == 1:
        Centroid = np.array(initialize_random(k))
    if stratergy == 2:
        Centroid = np.array(initialize_optimal(X1,k))
    Centroid_old = np.zeros(Centroid.shape)
    cluster = np.zeros(len(X1))
    err = euclideanDist(Centroid, Centroid_old, None)
    while err != 0:
        for i in range(len(X1)):
            distances = euclideanDist(X1[i], Centroid)
            clus = np.argmin(distances)
            cluster[i] = clus
        Centroid_old = deepcopy(Centroid)
        for i in range(k):
            points = [X1[j] for j in range(len(X1)) if cluster[j] == i]
            Centroid[i] = np.mean(points, axis=0)  
        err = euclideanDist(Centroid, Centroid_old, None)
    return Centroid,cluster
"""
Function for Plotting the graph
"""
def plot_graph(stratergy): 
    k_c = []
    obj_func =[]
    k_c1 = []
    obj_func1 =[]
    if stratergy == 1:
        for k in range(2, 11):    
            C,clusters = kmeans(k,1)
            k_c.append(k)
            obj_func.append(objectFunc(X1,C,clusters))
        plt.plot(k_c, obj_func)
        plt.scatter(k_c, obj_func)
        plt.xlabel('K Value')
        plt.ylabel('Objective Function')
        plt.title('Plot to find Optimal K value')
        plt.show()
    if stratergy == 2:
        for k1 in range(2, 11):    
            C1,clusters1 = kmeans(k1,2)
            k_c1.append(k1)
            obj_func1.append(objectFunc(X1,C1,clusters1))
            if k1 == 10:
                plt.scatter(X1[:, 0], X1[:, 1], marker = '.',  
                    color = 'chartreuse', label = 'Data') 
                plt.scatter(C1[:-1, 0], C1[:-1, 1],marker = '*', 
                            color = 'red') 
                plt.scatter(C1[-1, 0], C1[-1, 1],marker = '*',
                            color = 'red') 
                plt.legend() 
                plt.xlim(-2, 12) 
                plt.ylim(-2, 15) 
                plt.show()
        plt.plot(k_c1, obj_func1)
        plt.scatter(k_c1, obj_func1)
        plt.xlabel('K Value')
        plt.ylabel('Objective Function')
        plt.title('Plot to find Optimal K value')
        plt.show()
"""
Main Function
"""
def main():
    warnings.filterwarnings("ignore")
    print("K-means Using Stratergy 1")
    print("Random Initialization - 1")
    plot_graph(1)
    print("Random Initialization - 2")   
    plot_graph(1)
    print("K-means Using Stratergy 2")
    print("Initialization - 1")
    print("Plot to represent K = 10")
    plot_graph(2)
    print("Initialization - 2")  
    print("Plot to represent K = 10")
    plot_graph(2)

"""
Invoking main function
"""
main() 
