import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import scipy.stats
import time as t
import pickle


def associate(w,A,B):
    return cosine_similarity(w.reshape(1,-1),A).mean() - cosine_similarity(w.reshape(1,-1),B).mean()

def difference(X,Y,A,B):
    return np.sum([associate(X[i,:],A,B) for i in range(X.shape[0])]) - np.sum([associate(Y[i,:],A,B) for i in range(Y.shape[0])])

def effect_size(X,Y,A,B):
    delta_mean =  np.mean([associate(X[i,:],A,B) for i in range(X.shape[0])]) - np.mean([associate(Y[i,:],A,B) for i in range(Y.shape[0])])

    XY = np.concatenate((X,Y),axis=0)
    s = [associate(XY[i,:],A,B) for i in range(XY.shape[0])]

    std_dev = np.std(s,ddof=1)
    var = std_dev**2

    return delta_mean/std_dev, var

def inn(a_huge_key_list):
    L = len(a_huge_key_list)
    i = np.random.randint(0, L)
    return a_huge_key_list[i]


def sample_statistics(X,Y,A,B,num = 100):
    XY = np.concatenate((X,Y),axis=0)
   
    def inner_1(XY,A,B):
        X_test_idx = np.random.choice(XY.shape[0],X.shape[0],replace=False)
        Y_test_idx = np.setdiff1d(list(range(XY.shape[0])),X_test_idx)
        X_test = XY[X_test_idx,:]
        Y_test = XY[Y_test_idx,:]
        return difference(X_test,Y_test,A,B)
    
    s = [inner_1(XY,A,B) for i in range(num)]

    return np.mean(s), np.std(s,ddof=1)

def p_value(X,Y,A,B,num=100):
    m,s = sample_statistics(X,Y,A,B,num)
    d = difference(X,Y,A,B)
    p = 1 - scipy.stats.norm.cdf(d,loc = m, scale = s)
    return p

def ceat_meta(ceat_groups, test=1,N=10000):
    nm = "data/ceat/bert_weat.pickle"
    ceat_dict = pickle.load(open(nm,'rb'))

    e_lst = [] 
    v_lst = [] 

    for i in range(N):
        X = np.array([ceat_dict[wd][np.random.randint(0,len(ceat_dict[wd]))] for wd in ceat_groups[test-1][0]])
        Y = np.array([ceat_dict[wd][np.random.randint(0,len(ceat_dict[wd]))] for wd in ceat_groups[test-1][1]])
        A = np.array([ceat_dict[wd][np.random.randint(0,len(ceat_dict[wd]))] for wd in ceat_groups[test-1][2]])
        B = np.array([ceat_dict[wd][np.random.randint(0,len(ceat_dict[wd]))] for wd in ceat_groups[test-1][3]])
        e,v = effect_size(X,Y,A,B)
        e_lst.append(e)
        v_lst.append(v)

    e_ary = np.array(e_lst)
    w_ary = 1/np.array(v_lst)

    q1 = np.sum(w_ary*(e_ary**2))
    q2 = ((np.sum(e_ary*w_ary))**2)/np.sum(w_ary)
    q = q1 - q2

    df = N - 1

    if q>df:
        c = np.sum(w_ary) - np.sum(w_ary**2)/np.sum(w_ary)
        tao_square = (q-df)/c
        print("tao>0")
    else:
        tao_square = 0

    v_ary = np.array(v_lst)
    v_star_ary = v_ary + tao_square
    w_star_ary = 1/v_star_ary

    # calculate combiend effect size, variance
    pes = np.sum(w_star_ary*e_ary)/np.sum(w_star_ary)
    v = 1/np.sum(w_star_ary)

    # p-value
    z = pes/np.sqrt(v)
    # p_value = 1 - scipy.stats.norm.cdf(z,loc = 0, scale = 1)
    p_value = scipy.stats.norm.sf(z,loc = 0, scale = 1)


    return pes, p_value