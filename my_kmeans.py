from collections import defaultdict
from random import uniform
from math import sqrt

import numpy as np


def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2

    Returns a new point which is the center of all the points.
    """
    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center


def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers where `k` is the number of unique assignments.
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for key,points in new_means.items():
        centers.append(point_avg(points))

    return centers



def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point.
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    #对于每个点，找与其最近的一个中心点，并标记
    assignments = []
    for point in data_points:
        distances = calculateDist(point,centers).tolist()
        shortest_index = distances.index(min(distances))
        assignments.append(shortest_index)
    return assignments


#计算对象间的欧式距离
#默认计算多个对象与单个对象之间的欧式距离
def calculateDist(A, B, flag=0):
    if (flag == 0):
        return np.sqrt(np.sum((A - B)**2, axis=1))
    else:
        return np.sqrt(np.sum((A - B)**2))



'''def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    find the minimum and maximum for each coordinate, a range.
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]

            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    return centers'''


def k_means(dataset, k_points, iter=2):
    '''if k_points == None:
        k_points = generate_k(dataset, k)'''
    dataset_att = dataset[:,:-2]
    #dataset不带标签
    #dataset_att = dataset
    cluster_list = []
    k = len(k_points)
    if len(dataset) == k:
        for row in dataset:
            cluster_list.append(np.reshape(row,(1,len(row))))
    else:
        if iter !=1:
            assignments = assign_points(dataset_att, k_points)
            old_assignments = None
            while assignments != old_assignments:
                new_centers = update_centers(dataset_att, assignments)
                old_assignments = assignments
                assignments = assign_points(dataset_att, new_centers)  
        else:
            assignments = assign_points(dataset_att, k_points)
        
        for single_label in range(k):
            assignments = np.array(assignments)
            cluster_list.append(dataset[assignments == single_label, :]) 
    return  cluster_list