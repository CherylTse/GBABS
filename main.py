'''
Author: Cheryl Tse
Date: 2023-04-03 10:31:22
LastEditTime: 2023-12-09 15:11:10
Description: 
'''
import csv
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE,SMOTENC, BorderlineSMOTE
from lightgbm import LGBMClassifier
from imblearn.under_sampling import TomekLinks,EditedNearestNeighbours
from imblearn.metrics import geometric_mean_score
from matplotlib import pyplot as plt
from sklearn import svm, tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

import Bound_GBS
import radomSamp
import XIA_GBS,GBS_Imbalanced
import Bound_GBS_C
#import BLS

def fitClass(X_train, Y_train, X_test, Y_test, ClassMothed):
    if ClassMothed == 'SVM':
        clf = svm.SVC()
    elif ClassMothed == 'LR':
        clf = LogisticRegression()
    elif ClassMothed == 'BPNN':
        clf = MLPClassifier(hidden_layer_sizes=(5,),random_state=1) 
        #clf = MLPClassifier(random_state=1)
    elif ClassMothed == 'KNN':
        clf = KNeighborsClassifier(algorithm='ball_tree',n_neighbors=5)
    elif ClassMothed == 'DT':
        clf = tree.DecisionTreeClassifier(random_state=1)
    elif ClassMothed == 'XGBoost':
        #clf = XGBClassifier(objective="binary:logistic", random_state=1)
        clf = XGBClassifier(objective="multi:softprob", random_state=1)
    elif ClassMothed == 'RF':
        clf = RandomForestClassifier(random_state=1)
    elif ClassMothed == 'lightgbm':
        clf = LGBMClassifier(verbosity=-1)
    elif ClassMothed == 'NB':
        clf = GaussianNB()
    try:
        clf.fit(X_train, Y_train)
    except:
        return 0.0, 0.0,0.0
    Y_pred = clf.predict(X_test)

    # 计算混淆矩阵
    #tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    #cm = confusion_matrix(Y_test, Y_pred)
    # 计算每个类别的真正例率（召回率）
    #recall_per_class = np.diag(cm) / np.sum(cm, axis=1)
    #sensitivity = tp / (tp + fn)
    #specificity = tn / (tn + fp)
    
    #g_mean = np.sqrt(sensitivity * specificity)
    
    # 计算几何平均
    #g_mean = np.prod(recall_per_class) ** (1 / len(recall_per_class))
    try:
        g_mean = geometric_mean_score(Y_test, Y_pred, average='macro')
    except:
        g_mean = 0.0
    Accuracy = accuracy_score(Y_test, Y_pred)
    F1 = f1_score(Y_test, Y_pred, average='macro')
    #recall = recall_score(Y_test, Y_pred, average='macro')
    '''tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).flatten()
    tpr = tp / (tp + fn)  # 真阳性率（召回率）
    tnr = tn / (tn + fp)  # 真阴性率
    g_mean = (tpr * tnr) ** 0.5  # G-mean'''
    #g_mean=calculate_g_mean(Y_test, Y_pred, np.unique(Y_test))
    return  Accuracy,F1,g_mean


# 可视化采样数据
def plot_2(Sampdata):
    '''color = {1: 'k', -1: 'r'}
    plt.figure(figsize=(5, 4))
    plt.axis([0, 1, 0, 1])
    data0 = Sampdata[Sampdata[:, 0] == 1]
    data1 = Sampdata[Sampdata[:, 0] == -1]
    plt.plot(data0[:, 1], data0[:, 2], '.', color=color[1], markersize=5)
    plt.plot(data1[:, 1], data1[:, 2], '.', color=color[-1], markersize=5)'''
    num_classes = len(np.unique(Sampdata[:, 0]))
    # 设置颜色映射
    cmap = plt.cm.get_cmap('tab10', num_classes)

    # 绘制散点图
    plt.scatter(Sampdata[:, 1], Sampdata[:, 2], c=Sampdata[:, 0], cmap=cmap)
    plt.show()

# 可视化二维数据粒球
def plot_gb_2(granular_ball_list):
    color = {1: 'k', 0: 'r'}
    plt.figure(figsize=(5, 4))
    plt.axis([0, 1, 0, 1])
    for granular_ball in granular_ball_list:
        label = granular_ball.label
        center, radius = granular_ball.center, granular_ball.radius
        #取出每个球包含的样本，样本不一定都在球内
        data0 = granular_ball.data[granular_ball.data[:, 0] == 1]
        data1 = granular_ball.data[granular_ball.data[:, 0] == 0]
        plt.plot(data0[:, 1], data0[:, 2], '.', color=color[1], markersize=5)
        plt.plot(data1[:, 1], data1[:, 2], '.', color=color[0], markersize=5)
        #根据球的中心点及半径画圆，不是所有的样本均在圆内
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        plt.plot(x, y, color[label], linewidth=0.8)
        #画出圆心
        plt.plot(center[0], center[1], 'x', color=color[label])

    plt.show()




#以原始数据集作为input,先构造GB,再将GBs作为分类器的input
def main_ori(k=5,Noise_ratio = 0):
    warnings.filterwarnings("ignore")  # 忽略警告
    data_list = ['creditApproval','diabetes','car',
        'Pumpkin_Seeds_Dataset','banana','page-blocks',
        'coil2000','Dry_Bean_Dataset','HTRU2','magic']
    data_list = ['shuttle']
    #data_list = ['Dry_Bean_Dataset']
    purity = 1.0
    #k = 5 #基于密度的GBG方法中中心检测的超参数
    #Noise_ratio = 0.1  #标签噪声数据的比例
    repetitions = 5    #重复执行的次数 
    baseline = 'XGBoost'
    print('Noise_ratio:',Noise_ratio)
    print('noise dect co:',k)
    print('repetitions:',repetitions)
    #time1,time2 = 0.0,0.0
    for data_nm in data_list:
        flag = 0   #循环次数的标识
        print(data_nm)
        print('--------------------------------------------')
        #data_frame = pd.read_csv(r"E:\code\datasets\noise\noise" + str(Noise_ratio) + '/'+ data_nm + str(Noise_ratio) + ".csv",
        #                         header=None)  # 加载数据集
        data_frame = pd.read_csv(r"/home/cheryl/Desktop/GBC-1.0/datasets/noise" + str(Noise_ratio) + '/'+ data_nm + str(Noise_ratio) + ".csv",
                                 header=None)  # 加载数据集
        #data_frame = pd.read_csv("/Users/xieqin/Documents/code/GBC_NEW/datasets/"+ data_nm + ".csv",header=None)
        data = data_frame.values  #取出数据
        data = np.array(data)
        print(data.shape)
        print(Counter(data[:, 0]))
        numberSample = data.shape[0]
        #feature_num = data.shape[1]-1
        minMax = MinMaxScaler()
        data = np.hstack((data[:, 0].reshape(numberSample, 1),
                            minMax.fit_transform(data[:, 1:])))  #将属性列归一化
        train_data = data[:, 1:]  
        train_target = data[:, 0]  
        skf = StratifiedKFold(5, shuffle=True,random_state=1993)  
        RANDSacc_sum,BND_GBSacc_sum,BND_GBSf1_sum,basef1_sum = 0,0,0,0
        XIAGBSacc_sum,baseacc_sum,RANDSf1_sum,XIAGBSf1_sum = 0,0,0,0
        XIAsamp_rate_sum,bnd_gbsrate_sum= 0,0
        XIAIGBSacc_sum,XIAIGBSf1_sum,XIAIsamp_rate_sum = 0,0,0
        bnd_gb_samples_sum = 0
        while True:
            RANDSacc_sum_fold,BND_GBSacc_sum_fold,BND_GBSf1_sum_fold,basef1_sum_fold = 0,0,0,0
            XIAGBSacc_sum_fold,baseacc_sum_fold,RANDSf1_sum_fold,XIAGBSf1_sum_fold = 0,0,0,0
            XIAsamp_rate_sum_fold,bnd_gbsrate_sum_fold= 0,0
            XIAIGBSacc_sum_fold,XIAIsamp_rate_sum_fold,XIAIGBSf1_sum_fold = 0,0,0
            bnd_gb_samples_fold = 0
            fold = 0
            for train_index, test_index in skf.split(train_data, train_target):
                train, test = data[train_index], data[test_index]
                #train,test = train_test_split(data, test_size = 0.25,random_state = 1993)
                #print(train.shape)
                #plot_2(train)
                '''with open(str(data_nm)+str(Noise_ratio)+'test'+str(flag)+str(fold)+'.csv','w') as csvfile:
                    writer = csv.writer(csvfile)
                    for row in test:
                        writer.writerow(row)'''
                X_train = train[:,1:]
                Y_train = train[:,0]
                X_test = test[:,1:]
                Y_test = test[:,0]
                #（类边界+密度点）采样,在构建粒球的过程中识别类边界样本,同时将粒球的中心点作为密度点,进行采样(our1)
                #Bound_GBS
                train_num = len(train)
                train_withIndex = np.concatenate((train,np.array(range(train_num)).reshape(train_num,1)),axis=1)
                bnd_gb_samples, bnd_samples = Bound_GBS.bound_sampling(train_withIndex,k)
                bnd_gb_samples_fold += len(bnd_gb_samples)
                #print('num of sampling gbs in Bound_GBS:', len(bnd_gb_samples))
                bnd_gbs_rate = len(bnd_samples)/len(train)
                bnd_gbsrate_sum_fold += bnd_gbs_rate
                bnd_samples = np.array(bnd_samples)
                with open(str(data_nm)+str(Noise_ratio)+str(flag)+str(fold)+'.csv','w') as csvfile:
                    writer = csv.writer(csvfile)
                    for row in bnd_samples:
                        writer.writerow(row)
                #plot_gb_2(bnd_gb_samples)
                #plot_2(bnd_samples)
                BNDX = bnd_samples[:,1:]
                BNDY = bnd_samples[:,0]
                BND_GBSacc,BND_GBSf1 = fitClass(BNDX, BNDY, X_test, Y_test, baseline)[0:2]
                BND_GBSacc_sum_fold += BND_GBSacc
                BND_GBSf1_sum_fold += BND_GBSf1
                #随机采样，采样比例与本文提出的GBS保持一致
                randSampled_data = radomSamp.main(train,bnd_gbs_rate)
                #plot_2(randSampled_data)
                RANDX_train = randSampled_data[:,1:]
                RANDY_train = randSampled_data[:,0]
                RANDSacc,RANDSf1 = fitClass(RANDX_train, RANDY_train, X_test, Y_test, baseline)[0:2]
                RANDSacc_sum_fold += RANDSacc
                RANDSf1_sum_fold += RANDSf1
                #plot_gb_2(GB_List)

                #夏的采样方法
                #start3 = time.time()
                XIAGBSX_train, XIAGBSY_train, XIAGB_num = XIA_GBS.main(X_train, Y_train, purity)
                #print('num of gbs in XIAGBS:',XIAGB_num)
                #XIAGBSX = np.concatenate(np.array(XIAGBSY_train).reshape(len(XIAGBSY_train),1), np.array(XIAGBSX_train))
                XIAGBSX = np.hstack(( np.array(XIAGBSX_train),np.array(XIAGBSY_train).reshape(len(XIAGBSY_train),1)))   #为了画图而拼接
                #plot_2(XIAGBSX)
                with open('xia'+str(data_nm)+str(Noise_ratio)+str(flag)+str(fold)+'.csv','w') as csvfile:
                    writer = csv.writer(csvfile)
                    for row in XIAGBSX:
                        writer.writerow(row)
                XIAGBSacc,XIAGBSf1 = fitClass(XIAGBSX_train, XIAGBSY_train, X_test, Y_test, baseline)[0:2]
                XIAGBSacc_sum_fold += XIAGBSacc
                XIAGBSf1_sum_fold += XIAGBSf1
                #end3 = time.time()
                #基于夏的采样方法的采样率
                XIAsamp_rate_sum_fold += len(XIAGBSX_train)/len(X_train)
                #plot_2(train)
                
                #夏的不平衡采样方法
                XIAIGBSX_train, XIAIGBSY_train = GBS_Imbalanced.main(X_train, Y_train, purity)
                XIAIGBSX = np.hstack(( np.array(XIAIGBSX_train),np.array(XIAIGBSY_train).reshape(len(XIAIGBSY_train),1)))   #为了画图而拼接
                #plot_2(XIAGBSX)
                with open('xiaI'+str(data_nm)+str(Noise_ratio)+str(flag)+str(fold)+'.csv','w') as csvfile:
                    writer = csv.writer(csvfile)
                    for row in XIAIGBSX:
                        writer.writerow(row)
                XIAIGBSacc,XIAIGBSf1 = fitClass(XIAIGBSX_train, XIAIGBSY_train, X_test, Y_test, baseline)[0:2]
                XIAIGBSacc_sum_fold += XIAIGBSacc
                XIAIGBSf1_sum_fold += XIAIGBSf1                
                #基于夏的不平衡采样方法的采样率
                XIAIsamp_rate_sum_fold += len(XIAIGBSX_train)/len(X_train)

                #不做采样处理，直接将数据作为input放入分类算法
                baseacc,basef1 = fitClass(X_train, Y_train, X_test, Y_test, baseline)[0:2]
                baseacc_sum_fold += baseacc
                basef1_sum_fold += basef1
                fold+=1
            bnd_gb_samples_sum += bnd_gb_samples_fold
            RANDSacc_sum += RANDSacc_sum_fold
            BND_GBSacc_sum += BND_GBSacc_sum_fold
            BND_GBSf1_sum += BND_GBSf1_sum_fold
            basef1_sum += basef1_sum_fold
            XIAGBSacc_sum += XIAGBSacc_sum_fold
            XIAIGBSacc_sum += XIAIGBSacc_sum_fold
            baseacc_sum += baseacc_sum_fold
            RANDSf1_sum += RANDSf1_sum_fold
            XIAGBSf1_sum += XIAGBSf1_sum_fold
            XIAIGBSf1_sum += XIAIGBSf1_sum_fold
            XIAsamp_rate_sum += XIAsamp_rate_sum_fold
            XIAIsamp_rate_sum += XIAIsamp_rate_sum_fold
            bnd_gbsrate_sum += bnd_gbsrate_sum_fold
            '''print('avg fold BND_GBS sampling sample rate:', bnd_gbsrate_sum_fold/5)
            print('avg fold XIA sampling sample rate:', XIAsamp_rate_sum_fold/5)
            print('avg fold XIAI sampling sample rate:', XIAIsamp_rate_sum_fold/5)
            print('avg fold acc of BNDGBS:', BND_GBSacc_sum_fold/5)
            print('avg fold acc of XIAGBS:', XIAGBSacc_sum_fold/5)
            print('avg fold acc of XIAIGBS:', XIAIGBSacc_sum_fold/5)
            print('avg fold acc of RANDS:', RANDSacc_sum_fold/5)
            print('avg fold acc of ' + baseline +':' , baseacc_sum_fold/5)
            print('avg fold f1 of BNDGBS:', BND_GBSf1_sum_fold/5)
            print('avg fold f1 of XIAGBS:', XIAGBSf1_sum_fold/5)
            print('avg fold f1 of XIAIGBS:', XIAIGBSf1_sum_fold/5)
            print('avg fold f1 of RANDS:', RANDSf1_sum_fold/5)
            print('avg fold f1 of ' + baseline +':' , basef1_sum_fold/5)
            print('--------------------------------------------')'''
            flag += 1
            if flag >= repetitions:
                break
        print('avg fold BND_GBS sampling gbs:', bnd_gb_samples_sum/(flag*5))
        print('avg BND_GBS sampling sample rate:', bnd_gbsrate_sum/(flag*5))
        print('avg XIA sampling sample rate:', XIAsamp_rate_sum/(flag*5))
        print('avg XIAI sampling sample rate:', XIAIsamp_rate_sum/(flag*5))
        print('avg acc of BNDGBS:', BND_GBSacc_sum/(flag*5))
        print('avg acc of XIAGBS:', XIAGBSacc_sum/(flag*5))
        print('avg acc of XIAIGBS:', XIAIGBSacc_sum/(flag*5))
        print('avg acc of RANDS:', RANDSacc_sum/(flag*5))
        print('avg acc of ' + baseline +':' , baseacc_sum/(flag*5))
        print('avg f1 of BNDGBS:', BND_GBSf1_sum/(flag*5))
        print('avg f1 of XIAGBS:', XIAGBSf1_sum/(flag*5))
        print('avg f1 of XIAIGBS:', XIAIGBSf1_sum/(flag*5))
        print('avg f1 of RANDS:', RANDSf1_sum/(flag*5))
        print('avg f1 of ' + baseline +':' , basef1_sum/(flag*5))
        print('--------------------------------------------')

#直接以训练好的GBs作为分类器的input(不包含GBG过程)
#input 可以是GBABS训练得到的采样后集合,也可以是其他方法训练得到的采样集合
def main(Noise_ratio=0):
    warnings.filterwarnings("ignore")  # 忽略警告
    data_list = ['creditApproval','diabetes','car',
        'Pumpkin_Seeds_Dataset','banana','page-blocks',
        'coil2000','Dry_Bean_Dataset','HTRU2','magic']
    #data_list = ['coil2000','HTRU2','magic','shuttle','Dry_Bean_Dataset']
    #data_list = ['UKB_2000_L28_M178_F431','UKB_2000_L44_M181_F480']
    data_list = ['shuttle']
    #Noise_ratio = 0.2  #标签噪声数据的比例
    repetitions = 5    #重复执行的次数 
    baseline = 'XGBoost'
    print('Noise_ratio:',Noise_ratio)
    print('repetitions:',repetitions)
    path = '/home/cheryl/Desktop/GBC-1.0/datasets/noise'
    #path = 'E:/code/datasets/noise/noise'
    path_sampled = '/home/cheryl/Desktop/GBS/sampled/'
    #path_sampled = 'E:/code/GBS/output/BABS/'
    for data_nm in data_list:
        flag = 0   #循环次数的标识
        print(data_nm)
        print('--------------------------------------------')        
        data = pd.read_csv(path + str(Noise_ratio) +'/' + data_nm + str(Noise_ratio) + ".csv",
                                 header=None)  # 加载原始数据集
        data = data.values  #取出数据
        data = np.array(data)
        print(data.shape)
        print(Counter(data[:, 0]))
        numberSample = data.shape[0]
        #feature_num = data.shape[1]-1
        minMax = MinMaxScaler()
        data = np.hstack((data[:, 0].reshape(numberSample, 1),
                            minMax.fit_transform(data[:, 1:])))  #将属性列归一化
        train_data = data[:, 1:]  
        train_target = data[:, 0]  
        BND_GBSacc_sum,BND_GBSf1_sum= 0,0
        baseacc_sum,basef1_sum = 0,0
        XIAGBSacc_sum,XIAGBSf1_sum = 0,0
        XIAIGBSacc_sum,XIAIGBSf1_sum,XIAIg_mean_sum = 0,0,0
        RANDSacc_sum,RANDSf1_sum = 0,0
        baseg_mean_sum,XIAg_mean_sum,RANDg_mean_sum,BNDg_mean_sum = 0,0,0,0
        smg_mean_sum,bsmg_mean_sum,smncg_mean_sum,tlg_mean_sum,enng_mean_sum = 0,0,0,0,0
        skf = StratifiedKFold(5, shuffle=True,random_state=1993)    #五折交叉
        while True:
            fold = 0
            RANDSacc_sum_fold,BND_GBSacc_sum_fold,BND_GBSf1_sum_fold,basef1_sum_fold = 0,0,0,0
            XIAGBSacc_sum_fold,baseacc_sum_fold,RANDSf1_sum_fold,XIAGBSf1_sum_fold = 0,0,0,0
            baseg_mean_sum_fold,XIAg_mean_sum_fold,RANDg_mean_sum_fold,BNDg_mean_sum_fold = 0,0,0,0
            smg_mean_sum_fold,bsmg_mean_sum_fold,smncg_mean_sum_fold,tlg_mean_sum_fold,enng_mean_sum_fold = 0,0,0,0,0
            XIAIGBSacc_sum_fold,XIAIGBSf1_sum_fold,XIAIg_mean_sum_fold = 0,0,0
            for train_index, test_index in skf.split(train_data, train_target):
                train, test = data[train_index], data[test_index]
                #base_train = train_test_split(data, test_size = 0.25,random_state = 1993)[0]
                X_train = train[:,1:]
                Y_train = train[:,0]
                #直接读取已经训练好的采样后数据
                ourTrain = pd.read_csv(path_sampled + data_nm + str(Noise_ratio) + str(flag)+str(fold)+ ".csv",
                                        header=None)  # 加载采样后数据集
                xiaTrain = pd.read_csv(path_sampled + 'xia' +data_nm + str(Noise_ratio) + str(flag)+str(fold)+ ".csv",
                                        header=None)  # 加载采样后数据集
                xiaITrain = pd.read_csv(path_sampled + 'xiaI' +data_nm + str(Noise_ratio) + str(flag)+str(fold)+ ".csv",
                                        header=None)  # 加载采样后数据集
                #test = pd.read_csv(path_sampled + data_nm + str(Noise_ratio) + 'test' + str(flag)+str(fold)+ ".csv",
                #                    header=None)  # 加载测试数据集
                ourTrain = ourTrain.values  #取出数据
                ourTrain = np.array(ourTrain)
                xiaTrain = xiaTrain.values  #取出数据
                xiaTrain = np.array(xiaTrain)
                xiaITrain = xiaITrain.values  #取出数据
                xiaITrain = np.array(xiaITrain)                
                #test = test.values  #取出数据
                #test = np.array(test)
                #我们的模型
                ourX_train = ourTrain[:,1:]
                ourY_train = ourTrain[:,0]
                xiaX_train = xiaTrain[:,:-1]
                xiaY_train = xiaTrain[:,-1]
                xiaIX_train = xiaITrain[:,:-1]
                xiaIY_train = xiaITrain[:,-1]
                X_test = test[:,1:]
                Y_test = test[:,0]
                BND_GBSacc,BND_GBSf1,BNDg_mean = fitClass(ourX_train, ourY_train, X_test, Y_test, baseline)
                #BND_GBSacc,BND_GBSf1 = fitClass(ourX_train, ourY_train, X_test, Y_test, baseline)[:2]
                BND_GBSacc_sum_fold += BND_GBSacc
                BND_GBSf1_sum_fold += BND_GBSf1
                BNDg_mean_sum_fold += BNDg_mean
                bnd_gbs_rate = len(ourTrain)/len(X_train)
                #随机下采样
                randSampled_data = radomSamp.main(train,bnd_gbs_rate)
                #plot_2(randSampled_data)
                RANDX_train = randSampled_data[:,1:]
                RANDY_train = randSampled_data[:,0]
                RANDSacc,RANDSf1,RANDg_mean = fitClass(RANDX_train, RANDY_train, X_test, Y_test, baseline)
                RANDSacc_sum_fold += RANDSacc
                RANDSf1_sum_fold += RANDSf1
                RANDg_mean_sum_fold += RANDg_mean
                #夏的采样
                XIAGBSacc,XIAGBSf1,XIAg_mean = fitClass(xiaX_train, xiaY_train, X_test, Y_test, baseline)
                XIAGBSacc_sum_fold += XIAGBSacc
                XIAGBSf1_sum_fold += XIAGBSf1
                XIAg_mean_sum_fold += XIAg_mean
                #夏的不平衡采样方法
                #XIAIGBSX_train, XIAIGBSY_train = GBS_Imbalanced.main(X_train, Y_train, purity)
                #XIAIGBSX = np.hstack(( np.array(XIAIGBSX_train),np.array(XIAIGBSY_train).reshape(len(XIAIGBSY_train),1)))   #为了画图而拼接
                #plot_2(XIAGBSX)
                XIAIGBSacc,XIAIGBSf1,XIAIg_mean = fitClass(xiaIX_train, xiaIY_train, X_test, Y_test, baseline)
                XIAIGBSacc_sum_fold += XIAIGBSacc
                XIAIGBSf1_sum_fold += XIAIGBSf1
                XIAIg_mean_sum_fold += XIAIg_mean

                #不做采样处理，直接将数据作为input放入分类算法
                baseacc,basef1,baseg_mean = fitClass(X_train, Y_train, X_test, Y_test, baseline)
                baseacc_sum_fold += baseacc
                basef1_sum_fold += basef1 
                baseg_mean_sum_fold += baseg_mean
                '''#SMOTE非平衡数据采样方法
                sm = SMOTE(random_state=1)
                smX_res, smy_res = sm.fit_resample(X_train, Y_train)
                smg_mean = fitClass(smX_res, smy_res, X_test, Y_test, baseline)[-1]
                smg_mean_sum_fold += smg_mean
                #BorderlineSMOTE非平衡数据采样方法
                bsm = BorderlineSMOTE(random_state=1)
                bsmX_res, bsmy_res = bsm.fit_resample(X_train, Y_train)
                bsmg_mean = fitClass(bsmX_res, bsmy_res, X_test, Y_test, baseline)[-1]
                bsmg_mean_sum_fold += bsmg_mean
                #SMOTENC非平衡数据采样方法
                smnc = SMOTENC(random_state=1,categorical_features=[0])
                smncX_res, smncy_res = smnc.fit_resample(X_train, Y_train)
                smncg_mean = fitClass(smncX_res, smncy_res, X_test, Y_test, baseline)[-1]
                smncg_mean_sum_fold += smncg_mean
                #TomekLinks下采样
                TLinks = TomekLinks()
                tlX_res, tly_res = TLinks.fit_resample(X_train, Y_train)
                tlg_mean = fitClass(tlX_res, tly_res, X_test, Y_test, baseline)[-1]
                tlg_mean_sum_fold += tlg_mean   
                #EditedNearestNeighbours下采样     
                enn = EditedNearestNeighbours()
                ennX_res, enny_res = enn.fit_resample(X_train, Y_train)
                enng_mean = fitClass(ennX_res, enny_res, X_test, Y_test, baseline)[-1]
                enng_mean_sum_fold += enng_mean'''
                fold +=1
            RANDSacc_sum += RANDSacc_sum_fold
            BND_GBSacc_sum += BND_GBSacc_sum_fold
            BND_GBSf1_sum += BND_GBSf1_sum_fold
            basef1_sum += basef1_sum_fold
            XIAGBSacc_sum += XIAGBSacc_sum_fold
            XIAIGBSacc_sum += XIAIGBSacc_sum_fold
            baseacc_sum += baseacc_sum_fold
            RANDSf1_sum += RANDSf1_sum_fold
            XIAGBSf1_sum += XIAGBSf1_sum_fold
            XIAIGBSf1_sum += XIAIGBSf1_sum_fold
            baseg_mean_sum+=baseg_mean_sum_fold
            XIAg_mean_sum +=XIAg_mean_sum_fold
            XIAIg_mean_sum +=XIAIg_mean_sum_fold
            RANDg_mean_sum +=RANDg_mean_sum_fold
            BNDg_mean_sum +=BNDg_mean_sum_fold
            smg_mean_sum +=smg_mean_sum_fold
            smncg_mean_sum +=smncg_mean_sum_fold
            bsmg_mean_sum +=bsmg_mean_sum_fold
            tlg_mean_sum +=tlg_mean_sum_fold
            enng_mean_sum +=enng_mean_sum_fold
            print('avg fold acc of BNDGBS:', BND_GBSacc_sum_fold/5)
            print('avg fold acc of XIAGBS:', XIAGBSacc_sum_fold/5)
            print('avg fold acc of XIAIGBS:', XIAIGBSacc_sum_fold/5)
            print('avg fold acc of RANDS:', RANDSacc_sum_fold/5)
            print('avg fold acc of ' + baseline +':' , baseacc_sum_fold/5)
            print('avg fold f1 of BNDGBS:', BND_GBSf1_sum_fold/5)
            print('avg fold f1 of XIAGBS:', XIAGBSf1_sum_fold/5)
            print('avg fold f1 of XIAIGBS:', XIAIGBSf1_sum_fold/5)
            print('avg fold f1 of RANDS:', RANDSf1_sum_fold/5)
            print('avg fold f1 of ' + baseline +':' , basef1_sum_fold/5)
            print('avg fold g_mean of BNDGBS:', BNDg_mean_sum_fold/5)
            print('avg fold g_mean of XIAGBS:', XIAg_mean_sum_fold/5)
            print('avg fold g_mean of XIAIGBS:', XIAIg_mean_sum_fold/5)
            print('avg fold g_mean of RANDS:', RANDg_mean_sum_fold/5)
            print('avg fold g_mean of ' + baseline +':' , baseg_mean_sum_fold/5)
            print('--------------------------------------------')
            flag += 1
            if flag >= repetitions:
                break
        print('avg acc of BNDGBS:', BND_GBSacc_sum/(flag*5))
        print('avg acc of XIAGBS:', XIAGBSacc_sum/(flag*5))
        print('avg acc of XIAIGBS:', XIAIGBSacc_sum/(flag*5))
        print('avg acc of RANDS:', RANDSacc_sum/(flag*5))
        print('avg acc of ' + baseline +':' , baseacc_sum/(flag*5))
        print('avg f1 of BNDGBS:', BND_GBSf1_sum/(flag*5))
        print('avg f1 of XIAGBS:', XIAGBSf1_sum/(flag*5))
        print('avg f1 of XIAIGBS:', XIAIGBSf1_sum/(flag*5))
        print('avg f1 of RANDS:', RANDSf1_sum/(flag*5))
        print('avg f1 of ' + baseline +':' , basef1_sum/(flag*5))
        print('avg g_mean of BNDGBS:', BNDg_mean_sum/(flag*5))
        print('avg g_mean of XIAGBS:', XIAg_mean_sum/(flag*5))
        print('avg g_mean of XIAIGBS:', XIAIg_mean_sum/(flag*5))
        print('avg g_mean of RANDS:', RANDg_mean_sum/(flag*5))
        print('avg g_mean of SMOTE:', smg_mean_sum/(flag*5))
        print('avg g_mean of SMOTENC:', smncg_mean_sum/(flag*5))
        print('avg g_mean of TomekLinks:', tlg_mean_sum/(flag*5))
        print('avg g_mean of EditedNearestNeighbours:', enng_mean_sum/(flag*5))
        print('avg g_mean of BorderlineSMOTE:', bsmg_mean_sum/(flag*5))
        print('avg g_mean of ' + baseline +':' , baseg_mean_sum/(flag*5))
        print('--------------------------------------------')


if __name__ == '__main__':
    #参数敏感度测试
    '''Densitytole = []
    start = 3
    end = 19
    step = 2
    k = start
    while True:
        Densitytole.append(k)
        k += step
        if k > end:
            break
    for i in Densitytole:
        print('Densitytole:',i)
        main_ori(i)'''

    '''Noise_ratio = []
    start = 0.1
    end = 0.4
    step = 0.1
    k = start
    while True:
        Noise_ratio.append(np.round(k,1))
        k += step
        if k > end:
            break
    for i in Noise_ratio:
        #print('Noise_ratio:',Noise_ratio)
        main(i)'''
    #main_ori(5,0)
    main(0.3)
