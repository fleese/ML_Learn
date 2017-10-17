'''
Created on 2017年10月17日

@author: xiaojian1
'''
import pandas
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve,roc_auc_score,precision_score,recall_score,f1_score,precision_recall_curve

#自定义roc曲线  
def consum_roc_curve(data):
    #中文字体
    zh_font = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
    y = data.label         
    proba = data.probability  #为正样本的概率
    fpr_rt, tpr_rt, scores = roc_curve(y, proba)
    auc_score = roc_auc_score(y, proba)
    
    plt.plot(fpr_rt, tpr_rt, label='模型XX,auc_score=%.2f' % auc_score)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC 曲线',fontproperties=zh_font)
    plt.legend(loc='best',prop=zh_font)
    plt.show()


#计算recall,precision,f1
def recall_precision_f1(data):
    y = data.label 
    predict = data.predict
    prec_score = precision_score(y,predict)
    print("模型准确率:"+str(prec_score))
    rec_score = recall_score(y,predict)
    print("模型召回率:"+str(rec_score))
    f1 = f1_score(y,predict)
    print("模型f1:"+str(f1))


#正负样本概率分布
def probability_distribution(data):
    zh_font = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    pos_sample = data[data.label == 1] #所有原始正样本
    neg_sample = data[data.label == 0] #所有原始负样本
    
    ax1.hist(pos_sample.probability, bins=100, alpha=0.3, range=[0, 1], color='r')
    ax2.hist(neg_sample.probability, bins=100, alpha=0.3, range=[0, 1], color='b')
    
    plt.title("概率分布直方图",fontproperties=zh_font)
    plt.grid(True)
    ax1.set_ylabel('正样本数', color='r',fontproperties=zh_font)
    ax2.set_ylabel('负样本数', color='b',fontproperties=zh_font)
    ax1.set_xlabel("概率",fontproperties=zh_font)
    plt.show()

#概率分布曲线图(在同一坐标轴下)
def probability_distribution_curve(data):
    zh_font = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
    probas = [i/100 for i in range(0,100,1)]  #以0.01为步长的概率
    probas.append(1.0)
    
    pos_lens = []  #存放在某刻度范围内的正样本个数
    neg_lens = []  #存放在某刻度范围内的负样本个数
    
    pos_sample = data[data.label == 1] #所有原始正样本
    neg_sample = data[data.label == 0] #所有原始负样本
    
    for proba in probas:
        pos_lens.append(len(pos_sample[(pos_sample.probability >= proba) & (pos_sample.probability < proba+0.01)]))
        neg_lens.append(len(neg_sample[(neg_sample.probability >= proba) & (neg_sample.probability < proba+0.01)]))
           
    plt.plot(probas,pos_lens, label='正样本数')
    plt.plot(probas,neg_lens, label='负样本数')
    
    plt.title("概率分布曲线图",fontproperties=zh_font)
    plt.grid(True)
    plt.ylabel('样本数',fontproperties=zh_font)
    plt.xlabel("概率",fontproperties=zh_font)   
    plt.legend(loc='best',prop=zh_font)    
    plt.show()    
    
#p_r曲线
def pr_curve_present(data):
    zh_font = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
    precisions, recalls, thresholds   = precision_recall_curve(data.label,data.probability) 
    
    plt.plot(recalls, precisions, label='模型XX')
    plt.title("P-R曲线",fontproperties=zh_font)
    plt.grid(True)
    plt.xlabel("召回率",fontproperties=zh_font)
    plt.ylabel("准确率",fontproperties=zh_font)
    plt.legend(loc='best',prop = zh_font)
    plt.show()
    

#precision,recall以概率为X轴的独立曲线
def pr_separate_curve(data):
    zh_font = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
    precisions, recalls, thresholds   = precision_recall_curve(data.label,data.probability) 
    
    length = len(thresholds)
    new_p = precisions[0:length]
    new_r = recalls[0:length]
    
    plt.plot(thresholds, new_p,label='准确率')
    plt.plot(thresholds, new_r,label='召回率')
    
    plt.title("P-R曲线",fontproperties=zh_font)
    plt.grid(True)
    plt.legend(loc='best',prop=zh_font)  
    plt.show()
    
#正负样本概率累加分布曲线
def probability_distribution_addup(data):
    zh_font = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
    probas = [i/20 for i in range(0,20,1)] #以0.05为步长的概率数组
    probas.append(1.0)
    
    pos_lens = []
    neg_lens = [] 
    
    pos_sample = data[data.label == 1]
    neg_sample = data[data.label == 0]
    
    for proba in probas:
        pos_lens.append(len(pos_sample[pos_sample.probability >= proba]))
        neg_lens.append(len(neg_sample[neg_sample.probability >= proba]))
        
    plt.plot(probas,pos_lens, label='正样本数')
    plt.plot(probas, neg_lens, label='负样本数')
    
    plt.title("概率累计分布曲线图",fontproperties=zh_font)
    plt.grid(True)
    plt.ylabel('样本数',fontproperties=zh_font)
    plt.xlabel("概率",fontproperties=zh_font)   
    plt.legend(loc='best',prop=zh_font)    
    plt.show() 


if __name__ == '__main__':
    data = pandas.read_csv('../data/test_data.csv',delimiter='\t',
                          header=0,encoding='utf-8',index_col=None)
    
    #ROC 曲线
#     consum_roc_curve(data) 
    #计算recall,precision,f1
#     recall_precision_f1(data)
    #概率分布直方图
#     probability_distribution(data)  
    #概率分布曲线图,在同一坐标轴下
#     probability_distribution_curve(data)
    #p_r曲线
#     pr_curve_present(data)
    #precision,recall以概率为X轴的独立曲线
#     pr_separate_curve(data)
    #正负样本概率累加分布曲线
    probability_distribution_addup(data)
    
    