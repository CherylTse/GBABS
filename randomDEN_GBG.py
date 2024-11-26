'''
Author: Cheryl Tse
Date: 2023-04-06 19:54:00
LastEditTime: 2024-11-26 20:11:31
Description: 
'''


import math
from collections import Counter

import numpy as np


class GranularBall: 
    #粒球的一些基本属性
    def __init__(self, data) -> None:
        self.data = data
       # self.centroid = []
        self.center = []    #初始为空list，无影响
        self.radius = 0.0
        self.num = len(data)
        self.label = 0    #初始化

#计算粒球的质心
def calCentroid(data):
    Centroid = np.mean(data, axis=0)
    #dis = calculateDist(data, center)
    #radius = np.max(dis)
    return Centroid


#计算对象间的欧式距离
#默认计算多个对象与单个对象之间的欧式距离
def calculateDist(A, B, flag = 0):
    if (flag == 0):
        return np.sqrt(np.sum((A - B)**2, axis=1))
    else:
        return np.sqrt(np.sum((A - B)**2))
    
#计算样本到球面的距离
def calPo2ballsDis(point,GB_List):
    dis_list = []
    for ball in GB_List:
        dis = np.sqrt(np.sum((point[0][1:-1] - ball.center)**2)) - ball.radius     
        dis_list.append(dis)
    return dis_list
       

#计算样本到异类球面的距离
def calPo2heteballsDis(point,GB_List):
    dis_list = []
    for ball in GB_List:
        if ball.label != point[0][0]:
            dis = np.sqrt(np.sum((point[0][1:-1] - ball.center)**2)) - ball.radius
            dis_list.append(dis)
    #若无异类粒球，则将半径置为无穷大
    if len(dis_list) == 0:
        return math.inf
    else:
        return dis_list

#计算data中每个类别样本的均值向量作为聚类中心
def calCenters(data):
    center_list = []
    count = Counter(data[:, 0])
    labels = sorted(count, key=count.get,reverse=True)  #按类群由小到大排序,以使得后续划分粒球时，优先大群体
    #labels = np.unique(data[:,0])
    for label in labels:
        data_label = data[data[:,0]==label][:,1:]
        center = np.mean(data_label, axis=0)
        center_list.append([label,center])
    return center_list

#随机选择待划分样本中的异类中心点
def randCenters(data,lowDensity):
    center_list = []
    #tmp_data = data.copy()
    lowDensity = np.array(lowDensity)
    lowDensity_points = []
    len_lowDensity = len(lowDensity)
    if len_lowDensity > 1:
        lowDensity_points = lowDensity[:,0]
    elif len_lowDensity == 1:
        lowDensity_points = [lowDensity[0][0]]
    for index in lowDensity_points:
        if index in data[:,-1]:
            data = data[data[:,-1]!=index]   #删除低密度点
    count = Counter(data[:, 0])
    labels = sorted(count, key=count.get,reverse=True)  #按类群由小到大排序,以使得后续划分粒球时，优先大群体
    for label in labels:
        center_index = np.random.choice(data[data[:,0] == label].shape[0], size=1, replace=False) 
        center = data[data[:,0] == label][center_index]
        center_list.append(center)
    return center_list

#噪声检测
def OOD(dataDis_sort,k,center_label):
    k_dataDis_sort = dataDis_sort[:k,:]
    #k近邻中与center同标签的样本个数，除了center本身
    homo_num = len(k_dataDis_sort[k_dataDis_sort[:,0] == center_label][:,0]) -1   #去掉center本身
    #k近邻中无任何与center同标签的样本,则将center识别为outlier
    if homo_num == 0:
        return 0
    #k近邻中最近的样本是异类，但剩下的k-1个样本均与center同类
    elif homo_num == k-1:
        return 1
    else:
        return 2     #k近邻中除了最近的样本是异类，后续的k-1个样本中，不是所有样本都是同类，则将center视为低密度点

#对于待划分样本集合，根据规则构造粒球
def generateGBs(GB_List,lowDensity,data,k):
    GBs = []
    end_flag = 0
    tmp_GB_List = GB_List.copy()
    undo_data = np.array(data)
    #center_list = calCenters(data)
    center_list = randCenters(data,lowDensity)
    #若剩下的样本全是低密度中心点，则不再做任何处理
    if len(center_list) != 0: 
        for center in center_list:
            min_dis_ball = math.inf
            tmp_radius = math.inf
            center_label = center[0][0]
            #centerF = center[0][1:-1]   #去掉index
            #对于随机选择的中心做有效性检查。有效性：是否能形成不冲突且不包含噪声的粒球
            #第一步：根据center与待划分样本的距离，判断center是不是噪声点。
            # 若与center最近的前k个样本都是异类点，则center可认为是outlier，删除center，并continue
            # 若最近的点是异类，而后续的k-1个是与center同类的，则将最近的这个点识别为outlier，删除outlier，并进入中心有效性检查的第二步
            dis_list = np.array(calculateDist(center[0][1:-1], undo_data[:,1:-1]))
            data_dis = np.concatenate((undo_data, dis_list.reshape(len(dis_list),1)), axis=1)  #倒数第二列是样本的索引
            dataDis_sort = data_dis[data_dis[:,-1].argsort()]     #距离按升序
            heter_indexlist = np.argwhere(dataDis_sort[:,0] != center_label)
            #判断待划分样本是否仅单一类别
            if len(heter_indexlist)!=0:
                heter_index = heter_indexlist[0][0]
                if heter_index == 1:   #若最近的样本是异类样本(除了center本身)，则需要做噪声检测，并对该center不再做粒球构造流程
                    #print('test')
                    OOD_flag = OOD(dataDis_sort,k,center_label)
                    if OOD_flag == 0:   #center与周边的k近邻无同类，则将该center识别与噪声点，删除;这种也可以识别为overlap的数据，参考:[1]
                        undo_data = undo_data[undo_data[:,-1] != center[0][-1]]
                        #print('delete test1')
                        continue
                    elif OOD_flag == 2:
                        lowDensity.append([center[0][-1],0])    #记录下标签维度视角下的低密度点的样本索引
                        continue
                    else:
                        dataDis_sort.remove(dataDis_sort[1])  #最近的异类识别为outlier，删掉;这种也可以识别为overlap的数据，参考:[1]
                        #print('delete test2')
                        heter_indexlist = heter_indexlist[1:]
                        if len(heter_indexlist)!=0:
                            heter_index = heter_indexlist[0][0]
                            tmp_radius = dataDis_sort[heter_index-1][-1]
                        else:
                            tmp_radius = dataDis_sort[-1,-1]
                else:
                    tmp_radius = dataDis_sort[heter_index-1][-1]
            #待划分样本仅单一类别,则根据求其最近异类样本的距离为半径的方式无法构造粒球，暂时将其半径置为无穷大，待后面判断其到异类粒球的距离
            else:
                tmp_radius = dataDis_sort[-1,-1]
            #第二步：为了避免所构造的粒球与前序构造的粒球异类overlap(冲突)，则在构造新GB时，多一步，限定新GB的半径
            # 判断前序是否已经构造了粒球，若没有则min_dis_ball默认取无穷大
            if len(tmp_GB_List) !=0:
                min_dis_ball = np.min(calPo2ballsDis(center,tmp_GB_List))
            else:   #若当前还没有构造好的粒球，则将center到粒球的球面距离置为无穷大
                min_dis_ball = math.inf
            #若扩张时先遇到已划分的球或异类样本，则停止扩张，并构成新球
            #若待划分样本仅单一类别，且当前无异类粒球，则将所有待划分样本直接构造粒球            
            if tmp_radius <= min_dis_ball:
                radius = tmp_radius
                gb_data = dataDis_sort[:heter_index,:-1]
                undo_data = dataDis_sort[heter_index:,:-1]
            #若到异类粒球的距离小于center到最近异类样本的距离，则以<=异类粒球的距离的最远同类样本的距离作为半径构造粒球
            else:
                index_list = np.argwhere(dataDis_sort[:,-1] <= min_dis_ball)  #取小于到已有球的距离的所有与center同类的样本索引
                if len(index_list) <= 1:
                    lowDensity.append([center[0][-1],1])    #避免粒球overlap，而将这些会使得overlap的样本作为低密度点
                    continue
                else:
                    max_index = np.max(index_list)  #取小于到已有球的距离的最远样本的索引
                    radius = dataDis_sort[max_index][-1]
                    gb_data = dataDis_sort[:max_index+1,:-1]
                    undo_data = dataDis_sort[max_index+1:,:-1]            
            #装配粒球基本属性
            new_ball = GranularBall(gb_data)
            new_ball.radius = radius
            new_ball.center = center[0][1:-1]
            #new_ball.centroid = calCentroid(gb_data[:,1:-1])
            new_ball.label = center_label
            GBs.append(new_ball)
            tmp_GB_List.append(new_ball)
    else:
        end_flag = 1
    return GBs,undo_data,end_flag


#对于训练样本集合迭代地基于待划分样本集构造粒球，直到达到停止条件
#k表示噪声检测方法的超参数
def generateGBList(data,k):
    GB_List = []
    undo_data = data
    lowDensity = []
    flag = 0
    centerlist = []
    #centroidlist = []
    while True:
        GBs,undo_data,flag = generateGBs(GB_List,lowDensity,undo_data,k)
        #若剩下的样本不能再构造样本（未划分样本均是前序判断的低密度中心点），则停止
        if flag ==1:   #待划分样本均属于低密度点
            break
        if len(GBs) !=0:
            for gb in GBs:
                if gb.num > 0:
                    GB_List.append(gb)
                    centerlist.append(gb.center)
                    #centroidlist.append(gb.centroid)                    
        undo_category = np.unique(undo_data[:,0])
        #若待划分样本均无同类样本，则停止
        if len(undo_data) == len(undo_category):
            break
    '''#调整gb的几何结构
    for gb in GB_List:
        gb.center,gb.radius = calCenterRadius(gb.data[:,1:-1])
        centerlist.append(gb.center)'''
    #将所有待划分的样本分别构造为半径为0的粒球
    lowDensity = np.array(lowDensity)
    for sample_index in lowDensity[lowDensity[:,-1]==1][:,0]:
        if sample_index in undo_data[:,-1]:  
            gb = GranularBall(data[data[:,-1]==sample_index])
            gb.center = gb.data[0][1:-1]
            #gb.centroid = gb.data[0][1:-1]
            gb.label = gb.data[0][0]
            centerlist.append(gb.center)
            #centroidlist.append(gb.centroid)
            GB_List.append(gb)
            undo_data = undo_data[undo_data[:,-1]!=sample_index]
    return GB_List,centerlist

