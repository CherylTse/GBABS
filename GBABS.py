'''
Author: Cheryl Tse
Date: 2024-11-26 15:56:43
LastEditTime: 2024-11-26 20:09:16
Description: 
'''
import numpy as np
import RD_GBG


class GBABS:
    def __init__(self, data, rho):
        self.data = data
        self.rho = rho
        #self.bnd_gbs = [] # borderline GBs
        self.postsampling = []
        self.postsampling_index = []

    def bound_sampling(self):
        GB_list,centerlist = RD_GBG.generateGBList(self.data,self.rho)
        positionlist = np.array(centerlist)
        #在每一个特征维度上确定类边界上的异质balls，作为类边界粒球
        #i表示第i个特征
        for i in range(len(self.data[1])-2):
            #将所有center在第i个特征上的特征值进行排序，如升序
            i_feature_sort = np.argsort(positionlist[:,i])
            for j in range(len(i_feature_sort)-1):
                if GB_list[i_feature_sort[j]].label != GB_list[i_feature_sort[j+1]].label:    #在第i个特征上，相邻ball异质，则认为二者在该维度上处于类边界上
                    '''if GB_list[j] not in self.bnd_gbs:
                        self.bnd_gbs.append(GB_list[j])
                    if GB_list[j+1] not in self.bnd_gbs:
                        self.bnd_gbs.append(GB_list[j+1])'''
                    #根据二者的位置特性，分别将GB_list[j]在i特征上取最大的样本作为边界样本；
                    #反之，GB_list[j]在i特征上取最小的样本作为边界样本
                    tmp_sample1 = self.bound_ball_sampling(GB_list[j],i,1)
                    for example in tmp_sample1:
                        if example[-1] not in self.postsampling_index:
                            self.postsampling_index.append(example[-1])
                            self.postsampling.append(example[:-1])  #最后采样时，去掉样本索引
                    tmp_sample2 = self.bound_ball_sampling(GB_list[j+1],i,0)
                    for example in tmp_sample2:
                        if example[-1] not in self.postsampling_index:
                            self.postsampling_index.append(example[-1])
                            self.postsampling.append(example[:-1]) #最后采样时，去掉样本索引
        #return self.bnd_gbs,self.postsampling
        return self.postsampling

    #边界上的球，根据其与类边界的方向，采该方向上球内最近的样本
    #flag=1 表示feature维度的正向；flag=0，则表示负向
    #若多个样本在该维度上均能取到最小值或者最大值，则都采
    def bound_ball_sampling(self,ball,i,flag):
        if len(ball.data) == 1:
            return ball.data
        else:
            if flag == 1:   #在i维度上取特征值最大的样本
                i_max = np.max(ball.data[:,i+1])
                return ball.data[ball.data[:,i+1] == i_max]
            else:    #在i维度上取特征值最小的样本
                i_min = np.min(ball.data[:,i+1])
                return ball.data[ball.data[:,i+1] == i_min]
