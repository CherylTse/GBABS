'''
Author: Cheryl Tse
Date: 2023-11-24 09:58:55
LastEditTime: 2023-11-26 22:33:32
Description: 在夏老师的原始代码的基础上,增加了第64行,以避免对空类簇构造粒球
'''
from collections import Counter
from sklearn.cluster._kmeans import k_means
import numpy as np
import random
import my_kmeans

class GranularBall:
    def __init__(self, data):  # Data is labeled data, the penultimate column is label, and the last column is index
        self.data = data
        self.data_no_label = data[:, :-2]
        self.num, self.dim = self.data_no_label.shape  # Number of rows, number of columns
        self.center = self.data_no_label.mean(0)  # According to the calculation of row direction, the mean value of all the numbers in each column (that is, the center of the pellet) is obtained
        self.label, self.purity = self.__get_label_and_purity()  # The type and purity of the label to put back the pellet
        self.init_center = self.random_center()  # Get a random point in each tag
        self.label_num = len(set(data[:, -2]))
        self.boundaryData = None
        self.radius = None

    def random_center(self):
        """
            Function function: saving centroid
            Return: centroid of all generated clusters
        """
        center_array = np.empty(shape=[0, len(self.data_no_label[0, :])])
        for i in set(self.data[:, -2]):
            data_set = self.data_no_label[self.data[:, -2] == i, :]  # A label is equal to the set of all the points to label a point
            random_data = data_set[random.randrange(len(data_set)), :]  # A random point in the dataset
            center_array = np.append(center_array, [random_data], axis=0)  # Add to the line
        return center_array

    def __get_label_and_purity(self):
        """
           Function function: calculate purity and label type
       """
        count = Counter(self.data[:, -2])  # Counter, put back the number of class tags
        label = max(count, key=count.get)  # Get the maximum number of tags
        purity = count[label] / self.num  # Purity obtained, percentage of tags
        return label, purity

    def get_radius(self):
        """
           Function function: calculate radius
       """
        diffMat = np.tile(self.center, (self.num, 1)) - self.data_no_label
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        self.radius = distances.sum(axis=0) / self.num

    def split_clustering(self):
        """
           Function function: continue to divide the granule into several new granules
           Output: new pellet list
       """
        Clusterings = []
        ClusterLists = my_kmeans.k_means(dataset=self.data, k_points=self.init_center)
        for Cluster_data in ClusterLists:
            if len(Cluster_data) >=1:   #不需要空类簇
                single_cluster = np.array(Cluster_data)
                Cluster = GranularBall(single_cluster)
                Clusterings.append(Cluster)
        return Clusterings


    def getBoundaryData(self):
        """
           Function function: get the points (boundary points) that need to be sampled in the pellet
       """
        if self.dim * 2 >= self.num:
            self.boundaryData = self.data
            return
        boundaryDataFalse = np.empty(shape=[0, self.dim])
        boundaryDataTrue = np.empty(shape=[0, self.dim + 2])
        for i in range(self.dim):
            centdataitem = np.tile(self.center, (1, 1))
            centdataitem[:, i] = centdataitem[:, i] + self.radius
            boundaryDataFalse = np.vstack((boundaryDataFalse, centdataitem))
            centdataitem = np.tile(self.center, (1, 1))
            centdataitem[:, i] = centdataitem[:, i] - self.radius
            boundaryDataFalse = np.vstack((boundaryDataFalse, centdataitem))
        list_path = []
        for boundaryDataItem in boundaryDataFalse:
            diffMat = np.tile(boundaryDataItem, (self.num, 1)) - self.data_no_label
            sqDiffMat = diffMat ** 2
            sqDistances = sqDiffMat.sum(axis=1)
            distances = sqDistances ** 0.5
            sortedDistances = distances.argsort()
            for i in range(self.num):
                if (self.data[sortedDistances[i]][-1] not in list_path and self.data[sortedDistances[i]][-2] == self.label):
                    boundaryDataTrue = np.vstack((boundaryDataTrue, self.data[sortedDistances[i]]))
                    list_path.append(self.data[sortedDistances[i]][-1])
                    break
        self.boundaryData = boundaryDataTrue

