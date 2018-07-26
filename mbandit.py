# -*- coding: utf-8 -*-

from bandit_simulations import *
import numpy as np
from scipy import stats
from scipy.stats import beta
import matplotlib.pyplot as plt
import copy

# bandit计算函数
def mBandit(true_rewards,choice_func,beta_params = None):
    '''
    输入：
        是否点击的情况，矩阵类型 ,N*M
        M个主题，N个实验结果
    输出：
        beta函数的(win,lose)参数
    '''
    num_samples,K = true_rewards.shape
    
    if beta_params is not None :
    # seed the estimated params (to avoid )
        estimated_beta_params_func = copy.copy(beta_params)
    else:
        estimated_beta_params_func = np.zeros((K,2))
        prior_a = 1. # aka successes 
        prior_b = 1. # aka failures
        estimated_beta_params_func[:,0] += prior_a # allocating the initial conditions
        estimated_beta_params_func[:,1] += prior_b
    
    for i in range(0,num_samples):
        # pulling a lever & updating estimated_beta_params
        this_choice = choice_func(estimated_beta_params_func)

        # update parameters
        if true_rewards[i,this_choice] == 1:
            update_ind = 0
        else:
            update_ind = 1
            
        estimated_beta_params_func[this_choice,update_ind] += 1

    return estimated_beta_params_func



# 连续分布用概率密度函数，系统成功概率的分布情况
def BanditProbs(estimated_beta_params,topics,printf = False,plotf = False):
    x = np.arange(0.01,1,0.01) # np.linspace(0, 1, 100)
    probs = []
    for wins,lose in estimated_beta_params:
        y = stats.beta.pdf(x,wins,lose)     # 概率密度函数
        probs.append(x[y.argmax()])
        if plotf:
            plt.plot(x,y)
    if plotf:
        plt.xlabel('X : prob from 0 - 1')
        plt.title('probability density curve')
        plt.ylabel('Probability density')
        plt.legend(topics,loc='lower right')
        plt.show()
    if printf:
        for x,y in zip(topics,probs):
            print('【%s】 topic prob is : %s'%(x,y))
    return probs

# generator data
def generate_bandit_data(new_topic,topics):
    true_rewards = np.array([[i in item for i in topics] for item in new_topic])
    return true_rewards

if __name__ == '__main__':
    topics = ['news','sports','entertainment','edu','tech']
    top10 = [['sports','edu'],['tech','sports'],['tech','entertainment','edu'],['entertainment'],['sports','tech','sports']
             ,['edu'],['tech','news'],['tech','entertainment'],['tech'],['tech','edu']]
    
    ## 第一轮冷启动
    # generator data
    true_rewards = generate_bandit_data(top10,topics)
    # bandit model
    estimated_beta_params = mBandit(true_rewards,UCB)
    print('Cold boot ...')
    prob1 = BanditProbs(estimated_beta_params,topics,printf = True,plotf = True)
    
    
    # 第二轮 冷启动之上
    topic_POI = [['edu','news']]
    
    true_rewards = generate_bandit_data(topic_POI,topics)
    estimated_beta_params = mBandit(true_rewards,UCB,beta_params = estimated_beta_params)  # 加载之前的内容
    print(estimated_beta_params)
    print(' second start. ...')
    prob2 = BanditProbs(estimated_beta_params,topics,printf = True,plotf = True)

