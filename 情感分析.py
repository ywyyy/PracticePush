# 1, 准备数据集
import numpy as np
np.random.seed(37)
from nltk.corpus import movie_reviews
pos_fileIds=movie_reviews.fileids('pos') # 加载积极文本文件
neg_fileIds=movie_reviews.fileids('neg') # 消极文本文件 
print(len(pos_fileIds)) 
print(len(neg_fileIds))   
#print(pos_fileIds[:5])
#print(neg_fileIds[:5]) 
# 由此可看出，movie_reviews.fileids是加载各种类别文本的文件，
# 并返回该文件名组成的list 
# 如果想要查看某个文本文件的内容，可以使用
#print(movie_reviews.words(fileids=['pos/cv000_29590.txt']))





# 2, 处理数据集
def extract_features(word_list):
    #专门一个函数来提取特征
    return dict([(word,True) for word in word_list]) # 此处加True的作用是构成dict,实质意义不大
 
pos_features=[(extract_features(movie_reviews.words(fileids=[f])),'Pos') 
              for f in pos_fileIds]
neg_features=[(extract_features(movie_reviews.words(fileids=[f])),'Neg') 
              for f in neg_fileIds]
#print(pos_features[:3]) # 打印下看看内容是否正确 
dataset=pos_features+neg_features # 将两部分结合起来作为一个dataset




# 构建模型，训练模型
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
np.random.shuffle(dataset)
rows=int(len(dataset)*0.8) # 80%为train set
train_set,test_set=dataset[:rows],dataset[rows:]
print('Num of train_set: ',len(train_set),
      '/nNum of test_set: ',len(test_set))
clf=NaiveBayesClassifier.train(train_set)
 
# 查看该模型在test set上的表现
acc=nltk_accuracy(clf,test_set)
#acc=clf.prob_classify(test_set)
print('Accuracy: {:.2f}%'.format(acc*100))






# 用该模型来预测新样本，查看新句子的情感是积极还是消极
new_samples = [
        "It is an amazing movie", 
        "This is a dull movie. I would never recommend it to anyone.",
        "The cinematography is pretty great in this movie", 
        "The direction was terrible and the story was all over the place" 
    ]
 
for sample in new_samples:
   # p=extract_features(sample.split())
    #print(p)
    predict_P=clf.prob_classify(extract_features(sample.split()))
    pred_sentiment=predict_P.max()
    print('Sample: {}, Type: {}, Probability: {:.2f}%'.format(
        sample,pred_sentiment,predict_P.prob(pred_sentiment)*100))
