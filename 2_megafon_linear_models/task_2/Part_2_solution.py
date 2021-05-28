#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from scipy import stats

file = np.loadtxt('/Users/main/Мега_Задание_2/HW2_labels.txt',delimiter=',')
y_predict, y_true = file[:,:2], file[:,-1]
#print(y_predict.shape,y_true.shape)

#y_predict_sorted = -np.sort(-y_predict[:,1],axis = 0)
#print([round(i,3) for i in y_predict_sorted])

#percent = 0.5

#y_new = y_predict_sorted[0:int(y_predict.shape[0]*percent)]
#lowest_prob = y_new[len(y_new) - 1]
#print('lowest = ',lowest_prob)
#bool_a = (y_predict[:,1] > lowest_prob)*1
#print('bool_a.shape',bool_a.shape[0])
#print(bool_a)
    
#for i in range(0,10): print(bool_a[i])
#for i in range(0,10): print(int(y_true[i]))

#TP = (((bool_a - y_true) == 0)*(bool_a == 1)*1).sum()
#TN = (((bool_a - y_true) == 0)*(bool_a == 0)*1).sum()
#FP = ((abs(bool_a - y_true) == 1)*(bool_a == 1)*1).sum()
#FN = ((abs(bool_a - y_true) == 1)*(bool_a == 0)*1).sum()
#print('TP = ',TP)
#print('TN = ',TN)
#print('FP = ',FP)
#print('FN = ',FN)
#print('Summ = ',TP + TN + FP + FN)


def accuracy_score (y_true, y_predict, percent = None):
    y_predict_sorted = -np.sort(-y_predict[:,1], axis = 0)
    
    if percent is None:
        bool_a = (y_predict[:,1] > 0.5)*1
        TP = (((bool_a - y_true) == 0)*(bool_a == 1)*1).sum()
        TN = (((bool_a - y_true) == 0)*(bool_a == 0)*1).sum()
        
        result = (TP + TN)/y_true.shape[0]
        
        return result
    else:
        percent = percent/100
        if percent > 1 or percent <= 0:
            print('Результат неверный: 1 <= percent <= 100 нарушено')
            return None
      
    y_new = y_predict_sorted[0:int(y_predict.shape[0]*percent)]
    lowest_prob = y_new[len(y_new) - 1]
    bool_a = (y_predict[:,1] > lowest_prob)*1
    
    TP = (((bool_a - y_true) == 0)*(bool_a == 1)*1).sum()
    TN = (((bool_a - y_true) == 0)*(bool_a == 0)*1).sum()
    
    result = (TP + TN)/y_true.shape[0]
    
    return result

def precision_score (y_true, y_predict, percent = None):
    
    if percent is None:
        bool_a = (y_predict[:,1] > 0.5)*1
        TP = (((bool_a - y_true) == 0)*(bool_a == 1)*1).sum()
        FP = ((abs(bool_a - y_true) == 1)*(bool_a == 1)*1).sum()
        
        result = TP/(TP+FP)
        
        return result
    else:
        percent = percent/100
        if percent > 1 or percent <= 0:
            print('Результат неверный: 1 <= percent <= 100 нарушено')
            return None
    
    y_predict_sorted = -np.sort(-y_predict[:,1], axis = 0)
    y_new = y_predict_sorted[0:int(y_predict.shape[0]*percent)]
    lowest_prob = y_new[len(y_new) - 1]
    bool_a = (y_predict[:,1] > lowest_prob)*1
    
    TP = (((bool_a - y_true) == 0)*(bool_a == 1)*1).sum()
    FP = ((abs(bool_a - y_true) == 1)*(bool_a == 1)*1).sum()
    
    result = TP/(TP+FP)
    
    return result

def recall_score (y_true, y_predict, percent = None):
    y_predict_sorted = -np.sort(-y_predict[:,1], axis = 0)
    
    if percent is None:
        bool_a = (y_predict[:,1] > 0.5)*1
        TP = (((bool_a - y_true) == 0)*(bool_a == 1)*1).sum()
        FN = ((abs(bool_a - y_true) == 1)*(bool_a == 0)*1).sum()
        
        result = TP/(TP+FN)
        
        return result
    else:
        percent = percent/100
        if percent > 1 or percent <= 0:
            print('Результат неверный: 1 <= percent <= 100 нарушено')
            return None
        
    y_new = y_predict_sorted[0:int(y_predict.shape[0]*percent)]
    lowest_prob = y_new[len(y_new) - 1]
    bool_a = (y_predict[:,1] > lowest_prob)*1
    
    TP = (((bool_a - y_true) == 0)*(bool_a == 1)*1).sum()
    FN = ((abs(bool_a - y_true) == 1)*(bool_a == 0)*1).sum()
    
    result = TP/(TP+FN)
    
    return result

def lift_score (y_true, y_predict, percent = None):
    y_predict_sorted = -np.sort(-y_predict[:,1], axis = 0)
    
    if percent is None:
        bool_a = (y_predict[:,1] > 0.5)*1
        TP = (((bool_a - y_true) == 0)*(bool_a == 1)*1).sum()
        FN = ((abs(bool_a - y_true) == 1)*(bool_a == 0)*1).sum()
        
        prec = precision_score (y_true, y_predict)
        
        result = prec/((TP + FN)/y_true.shape[0])
        
        return result
    else:
        percent_ = percent/100
    y_new = y_predict_sorted[0:int(y_predict.shape[0]*percent_)]
    lowest_prob = y_new[len(y_new) - 1]
    bool_a = (y_predict[:,1] > lowest_prob)*1
    
    
    TP = (((bool_a - y_true) == 0)*(bool_a == 1)*1).sum()
    FN = ((abs(bool_a - y_true) == 1)*(bool_a == 0)*1).sum()
    
    prec = precision_score (y_true, y_predict, percent = percent_)
    
    result = prec/((TP + FN)/y_true.shape[0])
    return result

def f1_score (y_true, y_predict, percent = None): 
    rec = recall_score(y_true, y_predict, percent)
    prec = precision_score(y_true, y_predict, percent)
    result = 2*(rec * prec)/(rec + prec)
    return result

#acc = accuracy_score (y_true, y_predict,90)
#prec = precision_score (y_true, y_predict,90)
#rec = recall_score (y_true, y_predict,90)
#lift = lift_score (y_true, y_predict,90)
#f1 = f1_score (y_true, y_predict,90)

#print('acc = ', acc)
#print('prec = ', prec)
#print('rec = ', rec)
#print('lift = ',lift)
#print('f1 = ',f1)


# In[ ]:




