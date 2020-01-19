import pickle
import logging
from process_text import del_stopwords
import numpy as np
import pandas as pd
np.random.seed(1337)  


from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

# 创建词语字典，并返回word2vec模型中词语的索引，词向量
def create_dictionaries(p_model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(p_model.wv.vocab.keys(), allow_update=True)

    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: model[word] for word in w2indx.keys()}  # 词语的词向量
    return w2indx, w2vec


try:
    model = Word2Vec.load('./words2.model')

    # 索引字典、词向量字典
    index_dict, word_vectors= create_dictionaries(model)


    # 存储为pkl文件
    pkl_name = input("请输入保存的pkl文件名...\n")
    output = open(pkl_name + u".pkl", 'wb')
    pickle.dump(index_dict, output)  # 索引字典
    pickle.dump(word_vectors, output)  # 词向量字典
    output.close()

except Exception as e:
    print(e)
    # 主程序
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



    with open("./test_data/neg.txt", "r", encoding='UTF-8') as e:
        neg_data1 = e.readlines()


    with open("./test_data/pos.txt", "r", encoding='UTF-8') as s:
        pos_data1 = s.readlines()

    neg_data = sorted(set(neg_data1), key=neg_data1.index)  #列表去重 保持原来的顺序
    pos_data = sorted(set(pos_data1), key=pos_data1.index)


    neg_data = [del_stop_words(data.replace("\n", "")) for data in neg_data]
    pos_data = [del_stop_words(data.replace("\n", "")) for data in pos_data]
    data = neg_data + pos_data
    # a = 0
    # for i in data:
    #     a = a + len(i)
    # lang = a / len(data)
    # print(lang)   #打印每个列表平均长度
    # print(data)

    # c = ""
    # for i in data:
    #     for s in i:
    #         c = c + " " + s
    # c = c.replace("\n", "")
    # print(c)


    sentences = data  # 获取句子列表，每个句子又是词汇的列表

    print('训练Word2vec模型（可尝试修改参数）...')

    model = Word2Vec(sentences,
                     size=150,  # 词向量维度
                     min_count=1,  # 词频阈值
                     window=5)  # 窗口大小

    model_name = input("请输入保存的模型文件名...\n")
    model.save(model_name + '.model')  # 保存模型

if __name__ == "__main__":
    pass
