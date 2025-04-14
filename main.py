from collections import Counter
import warnings
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE,SMOTENC, BorderlineSMOTE
from lightgbm import LGBMClassifier
from imblearn.under_sampling import TomekLinks,EditedNearestNeighbours
from imblearn.metrics import geometric_mean_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

import GBABS
import radomSamp
import XIA_GBS,GBS_Imbalanced

def fitClass(X_train, Y_train, X_test, Y_test, ClassMothed):
    if ClassMothed == 'KNN':
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
    try:
        clf.fit(X_train, Y_train)
    except:
        return 0.0, 0.0,0.0
    Y_pred = clf.predict(X_test)

    try:  
        g_mean = geometric_mean_score(Y_test, Y_pred, average='macro')
    except:
        g_mean = 0.0
    Accuracy = accuracy_score(Y_test, Y_pred)
    return  Accuracy,g_mean


def main(rho=5, Noise_ratio = 0):
    warnings.filterwarnings("ignore") 
    data_list = ['creditApproval']
    purity = 1.0  # Purity threshold for Xia's sampling method
    repetitions = 5    # Number of repetitions for each dataset
    baseline = 'DT'
    print('Noise_ratio:',Noise_ratio)
    print('Density Tolerance:',rho)
    print('repetitions:',repetitions)
    for data_nm in data_list:
        flag = 0   
        print(data_nm)
        print('--------------------------------------------')
        data_frame = pd.read_csv(r"./datasets/noise" + str(Noise_ratio) + '/'+ data_nm + str(Noise_ratio) + ".csv",
                             header=None)  
        data = data_frame.values 
        data = np.array(data)
        print(data.shape)
        print(Counter(data[:, 0]))
        numberSample = data.shape[0]
        minMax = MinMaxScaler()
        data = np.hstack((data[:, 0].reshape(numberSample, 1),
                            minMax.fit_transform(data[:, 1:])))  
        train_data = data[:, 1:]  ## Features
        train_target = data[:, 0]  ## Labels
        skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=1993)  
        RANDS_acc_sum,RANDS_gmean_sum,GBABS_acc_sum,GBABS_gmean_sum = 0,0,0,0
        XIAGBS_acc_sum,XIAGBS_gmean_sum,XIAIGBS_acc_sum,XIAIGBS_gmean_sum,base_acc_sum,base_gmean_sum = 0,0,0,0,0,0
        XIAsamp_rate_sum,gbabs_rate_sum,XIAIsamp_rate_sum = 0,0,0
        smg_mean_sum,bsmg_mean_sum,smncg_mean_sum,tlg_mean_sum,enng_mean_sum = 0,0,0,0,0
        while True:
            RANDS_acc_fold,RANDS_gmean_fold,GBABS_acc_fold,GBABS_gmean_fold = 0,0,0,0
            XIAIGBS_acc_fold,XIAIGBS_gmean_fold,base_acc_fold,base_gmean_fold = 0,0,0,0
            XIAGBS_acc_fold,XIAGBS_gmean_fold = 0,0
            gbabs_rate_fold,XIAsamp_rate_fold,XIAIsamp_rate_fold = 0,0,0
            smg_mean_fold,bsmg_mean_fold,smncg_mean_fold,tlg_mean_fold,enng_mean_fold = 0,0,0,0,0
            fold = 0
            for train_index, test_index in skf.split(train_data, train_target):
                train, test = data[train_index], data[test_index]
                X_train = train[:,1:]
                Y_train = train[:,0]
                X_test = test[:,1:]
                Y_test = test[:,0]
                #GBABS
                train_num = len(train)
                train_withIndex = np.concatenate((train,np.array(range(train_num)).reshape(train_num,1)),axis=1)
                gbabs_instance = GBABS.GBABS(train_withIndex, rho)
                bnd_samples = gbabs_instance.bound_sampling()
                gbabs_rate = len(bnd_samples)/len(train)  
                gbabs_rate_fold += gbabs_rate
                bnd_samples = np.array(bnd_samples)
                # saving sampling data for each fold
                '''with open(str(data_nm)+str(Noise_ratio)+str(flag)+str(fold)+'.csv','w') as csvfile: 
                    writer = csv.writer(csvfile)
                    for row in bnd_samples:
                        writer.writerow(row)'''
                BNDX = bnd_samples[:,1:]
                BNDY = bnd_samples[:,0]
                GBABS_acc,GBABS_gmean = fitClass(BNDX, BNDY, X_test, Y_test, baseline)
                GBABS_acc_fold += GBABS_acc
                GBABS_gmean_fold += GBABS_gmean
                #Random sampling, the sampling ratio is consistent with GBABS
                randSampled_data = radomSamp.main(train,gbabs_rate)
                RANDX_train = randSampled_data[:,1:]
                RANDY_train = randSampled_data[:,0]
                RANDS_acc,RANDS_gmean = fitClass(RANDX_train, RANDY_train, X_test, Y_test, baseline)
                RANDS_acc_fold += RANDS_acc
                RANDS_gmean_fold += RANDS_gmean

                #Xia's sampling method, GBS
                XIAGBSX_train, XIAGBSY_train = XIA_GBS.main(X_train, Y_train, purity)[0:2]
                '''XIAGBSX = np.hstack(( np.array(XIAGBSX_train),np.array(XIAGBSY_train).reshape(len(XIAGBSY_train),1))) 
                with open('xia'+str(data_nm)+str(Noise_ratio)+str(flag)+str(fold)+'.csv','w') as csvfile:
                    writer = csv.writer(csvfile)
                    for row in XIAGBSX:
                        writer.writerow(row)'''
                XIAGBS_acc,XIAGBS_gmean = fitClass(XIAGBSX_train, XIAGBSY_train, X_test, Y_test, baseline)
                XIAGBS_acc_fold += XIAGBS_acc
                XIAGBS_gmean_fold += XIAGBS_gmean
                XIAsamp_rate_fold += len(XIAGBSX_train)/len(X_train)
                
                #Xia's sampling method for imbalanced datasets, GBS_Imbalanced
                XIAIGBSX_train, XIAIGBSY_train = GBS_Imbalanced.main(X_train, Y_train, purity)
                '''XIAIGBSX = np.hstack(( np.array(XIAIGBSX_train),np.array(XIAIGBSY_train).reshape(len(XIAIGBSY_train),1)))   
                with open('xiaI'+str(data_nm)+str(Noise_ratio)+str(flag)+str(fold)+'.csv','w') as csvfile:
                    writer = csv.writer(csvfile)
                    for row in XIAIGBSX:
                        writer.writerow(row)'''
                XIAIGBS_acc,XIAIGBS_gmean = fitClass(XIAIGBSX_train, XIAIGBSY_train, X_test, Y_test, baseline)
                XIAIGBS_acc_fold += XIAIGBS_acc
                XIAIGBS_gmean_fold += XIAIGBS_gmean                
                XIAIsamp_rate_fold += len(XIAIGBSX_train)/len(X_train)

                #training classifers without sampling
                base_acc,base_gmean = fitClass(X_train, Y_train, X_test, Y_test, baseline)
                base_acc_fold += base_acc
                base_gmean_fold += base_gmean
                #SMOTE
                sm = SMOTE(random_state=1)
                smX_res, smy_res = sm.fit_resample(X_train, Y_train)
                smg_mean = fitClass(smX_res, smy_res, X_test, Y_test, baseline)[-1]
                smg_mean_fold += smg_mean
                #BorderlineSMOTE
                bsm = BorderlineSMOTE(random_state=1)
                bsmX_res, bsmy_res = bsm.fit_resample(X_train, Y_train)
                bsmg_mean = fitClass(bsmX_res, bsmy_res, X_test, Y_test, baseline)[-1]
                bsmg_mean_fold += bsmg_mean
                #SMOTENC
                smnc = SMOTENC(random_state=1,categorical_features=[0])
                smncX_res, smncy_res = smnc.fit_resample(X_train, Y_train)
                smncg_mean = fitClass(smncX_res, smncy_res, X_test, Y_test, baseline)[-1]
                smncg_mean_fold += smncg_mean
                #TomekLinks
                TLinks = TomekLinks()
                tlX_res, tly_res = TLinks.fit_resample(X_train, Y_train)
                tlg_mean = fitClass(tlX_res, tly_res, X_test, Y_test, baseline)[-1]
                tlg_mean_fold += tlg_mean   
                #EditedNearestNeighbours  
                enn = EditedNearestNeighbours()
                ennX_res, enny_res = enn.fit_resample(X_train, Y_train)
                enng_mean = fitClass(ennX_res, enny_res, X_test, Y_test, baseline)[-1]
                enng_mean_fold += enng_mean
                fold+=1
            
            gbabs_rate_sum += gbabs_rate_fold
            XIAsamp_rate_sum += XIAsamp_rate_fold
            XIAIsamp_rate_sum += XIAIsamp_rate_fold
            GBABS_acc_sum += GBABS_acc_fold
            GBABS_gmean_sum += GBABS_gmean_fold
            RANDS_acc_sum += RANDS_acc_fold
            RANDS_gmean_sum += RANDS_gmean_fold
            XIAGBS_acc_sum += XIAGBS_acc_fold
            XIAGBS_gmean_sum += XIAGBS_gmean_fold
            XIAIGBS_acc_sum += XIAIGBS_acc_fold
            XIAIGBS_gmean_sum += XIAIGBS_gmean_fold
            base_acc_sum += base_acc_fold
            base_gmean_sum += base_gmean_fold
            smg_mean_sum += smg_mean_fold
            smncg_mean_sum += smncg_mean_fold
            bsmg_mean_sum += bsmg_mean_fold
            tlg_mean_sum += tlg_mean_fold
            enng_mean_sum += enng_mean_fold
            '''print('avg fold GBABS sampling sample rate:', gbabs_rate_fold/5)
            print('avg fold XIA sampling sample rate:', XIAsamp_rate_fold/5)
            print('avg fold XIAI sampling sample rate:', XIAIsamp_rate_fold/5)
            print('avg fold acc of GBABS:', GBABS_acc_fold/5)
            print('avg fold acc of XIAGBS:', XIAGBS_acc_fold/5)
            print('avg fold acc of XIAIGBS:', XIAIGBS_acc_fold/5)
            print('avg fold acc of RANDS:', RANDS_acc_fold/5)
            print('avg fold acc of ' + baseline +':' , base_acc_fold/5)
            print('avg fold gmean of GBABS:', GBABS_gmean_fold/5)
            print('avg fold gmean of XIAGBS:', XIAGBS_gmean_fold/5)
            print('avg fold gmean of XIAIGBS:', XIAIGBS_gmean_fold/5)
            print('avg fold gmean of RANDS:', RANDS_gmean_fold/5)
            print('avg fold gmean of ' + baseline +':' , base_gmean_fold/5)
            print('--------------------------------------------')'''
            flag += 1
            if flag >= repetitions:  
                break

        print('avg GBABS sampling sample rate:', gbabs_rate_sum/(flag*5))
        print('avg XIA sampling sample rate:', XIAsamp_rate_sum/(flag*5))
        print('avg XIAI sampling sample rate:', XIAIsamp_rate_sum/(flag*5))
        print('avg acc of GBABS:', GBABS_acc_sum/(flag*5))
        print('avg acc of XIAGBS:', XIAGBS_acc_sum/(flag*5))
        print('avg acc of XIAIGBS:', XIAIGBS_acc_sum/(flag*5))
        print('avg acc of RANDS:', RANDS_acc_sum/(flag*5))
        print('avg acc of ' + baseline +':' , base_acc_sum/(flag*5))
        print('avg gmean of GBABS:', GBABS_gmean_sum/(flag*5))
        print('avg gmean of XIAGBS:', XIAGBS_gmean_sum/(flag*5))
        print('avg gmean of XIAIGBS:', XIAIGBS_gmean_sum/(flag*5))
        print('avg gmean of RANDS:', RANDS_gmean_sum/(flag*5))
        print('avg gmean of ' + baseline +':' , base_gmean_sum/(flag*5))
        print('avg g_mean of SMOTE:', smg_mean_sum/(flag*5))
        print('avg g_mean of SMOTENC:', smncg_mean_sum/(flag*5))
        print('avg g_mean of TomekLinks:', tlg_mean_sum/(flag*5))
        print('avg g_mean of EditedNearestNeighbours:', enng_mean_sum/(flag*5))
        print('avg g_mean of BorderlineSMOTE:', bsmg_mean_sum/(flag*5))
        print('--------------------------------------------')

if __name__ == '__main__':
    # Parameter sensitivity analysis
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
        main(rho=i)'''

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
        #print('Noise_ratio:',i)
        main(Noise_ratio=i)'''
    main(5,0)
