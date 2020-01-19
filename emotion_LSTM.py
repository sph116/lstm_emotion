"""
功能：利用词向量+LSTM进行文本分类
"""

import numpy as np
np.random.seed(1337)  # For Reproducibility
import pickle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from process_text import del_stopwords
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt   #可视化类
from keras.models import load_model
import keras



# 参数设置
vocab_dim = 100  # 向量维度
maxlen = 150 # 文本保留的最大长度
batch_size = 100
n_epoch = 4
input_length = 150

f = open("./model/评价语料索引及词向量2.pkl", 'rb')  # 预先训练好的
index_dict = pickle.load(f)  # 索引字典，{单词: 索引数字}
word_vectors = pickle.load(f)  # 词向量, {单词: 词向量(100维长的数组)}

def show_train_history(train_history,train, velidation):
    """
    可视化训练过程 对比
    :param train_history:
    :param train:
    :param velidation:
    :return:
    """
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[velidation])
    plt.title("Train History")   #标题
    plt.xlabel('Epoch')    #x轴标题
    plt.ylabel(train)  #y轴标题
    plt.legend(['train', 'test'], loc='upper left')  #图例 左上角
    plt.show()

def text_to_index_array(p_new_dic, p_sen):  # 文本转为索引数字模式
    """
    文本或列表转换为索引数字
    :param p_new_dic:
    :param p_sen:
    :return:
    """
    if type(p_sen) == list:
        new_sentences = []
        for sen in p_sen:
            new_sen = []
            for word in sen:
                try:
                    new_sen.append(p_new_dic[word])  # 单词转索引数字
                except:
                    new_sen.append(0)  # 索引字典里没有的词转为数字0
            new_sentences.append(new_sen)
        return np.array(new_sentences)
    else:
        new_sentences = []
        sentences = []
        p_sen = p_sen.split(" ")
        for word in p_sen:
            try:
                sentences.append(p_new_dic[word])  # 单词转索引数字
            except:
                sentences.append(0)  # 索引字典里没有的词转为数字0
        new_sentences.append(sentences)
        return new_sentences

    # else:
    #     print(p_sen)
    #     p_sen = p_sen.split(" ")
    #     new_sentences = []
    #     for sen in p_sen:
    #         try:
    #             new_sentences.append(p_new_dic[word])  # 单词转索引数字
    #         except:
    #             new_sentences.append(0)  # 索引字典里没有的词转为数字0








# 定义网络结构
def train_lstm(p_n_symbols, p_embedding_weights, p_X_train, p_y_train, p_X_test, p_y_test, X_test_l):
    print('创建模型...')
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=p_n_symbols,
                        mask_zero=True,
                        weights=[p_embedding_weights],
                        input_length=input_length))

    model.add(LSTM(output_dim=100,
                   activation='sigmoid',
                   inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(units=512,
                    activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1,  # 输出层1个神经元 1代表正面 0代表负面
                    activation='sigmoid'))
    model.summary()

    print('编译模型...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("训练...")
    train_history = model.fit(p_X_train, p_y_train, batch_size=batch_size, nb_epoch=n_epoch,
              validation_data=(p_X_test, p_y_test))

    print("评估...")
    score, acc = model.evaluate(p_X_test, p_y_test, batch_size=batch_size)
    label = model.predict(p_X_test)
    print('Test score:', score)
    print('Test accuracy:', acc)
    for (a, b, c) in zip(p_y_test, X_test_l, label):
        print("原文为："+ "".join(b))
        print("预测倾向为", a)
        print("真实倾向为", c)

    show_train_history(train_history, 'acc', 'val_acc')  # 训练集准确率与验证集准确率 折线图
    show_train_history(train_history, 'loss', 'val_loss')  # 训练集误差率与验证集误差率 折线图

    """保存模型"""
    model.save('./model/emotion_model_LSTM.h5')
    print("模型保存成功")

def prodection(new_dic, test_txt):
    # 读取大语料文本


    """加载训练模型"""
    keras.backend.clear_session()
    model = load_model('./model/emotion_model_LSTM.h5')

    test_txt = del_stopwords(test_txt)

    test_txt = text_to_index_array(new_dic, test_txt)

    test_txt = sequence.pad_sequences(test_txt, maxlen=maxlen)


    predect = model.predict_classes(test_txt)
    # print(predect)
    item = {0: "负面", 1: "正面"}
    print(item[predect[0][0]])
    # return predect[0][0]


if __name__ == "__main__":

    try:
        model = load_model('./model/emotion_model_LSTM.h5')
        # 读取大语料文本

        new_dic = index_dict

        test_txt = input("请输入要预测的内容")
        prodection(new_dic, test_txt)

    except Exception as e:
        print(e)
        
        print("Setting up Arrays for Keras Embedding Layer...")
        n_symbols = len(index_dict) + 1  # 索引数字的个数，因为有的词语索引为0，所以+1
        embedding_weights = np.zeros((n_symbols, 100))  # 创建一个n_symbols * 100的0矩阵


        for w, index in index_dict.items():  # 从索引为1的词语开始，用词向量填充矩阵
            embedding_weights[index, :] = word_vectors[w]  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）


        # 读取语料分词文本，转为句子列表（句子为词汇的列表）
        with open("./原始语料/neg.txt", "r", encoding='UTF-8') as f:
            neg_data1 = f.readlines()

        with open("./原始语料/pos.txt", "r", encoding='UTF-8') as g:
            pos_data1 = g.readlines()

        neg_data = sorted(set(neg_data1), key=neg_data1.index)  #列表去重 保持原来的顺序
        pos_data = sorted(set(pos_data1), key=pos_data1.index)

        neg_data = [del_stopwords(data) for data in neg_data]
        pos_data = [del_stopwords(data) for data in pos_data]
        data = neg_data + pos_data


        # 读取语料类别标签
        label_list = ([0] * len(neg_data) + [1] * len(pos_data))


        # 划分训练集和测试集，此时都是list列表
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(data, label_list, test_size=0.2)

        # 转为数字索引形式

        # token = Tokenizer(num_words=3000)   #字典数量
        # token.fit_on_texts(train_text)

        X_train = text_to_index_array(index_dict, X_train_l)
        X_test = text_to_index_array(index_dict, X_test_l)

        print("训练集shape： ", X_train.shape)
        print("测试集shape： ", X_test.shape)


        y_train = np.array(y_train_l)  # 转numpy数组
        y_test = np.array(y_test_l)



        # 将句子截取相同的长度maxlen，不够的补0
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)

        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)
        # print(type(X_train), X_train[0: 5])
        print(type(y_train), y_train[0: 5])


        train_lstm(n_symbols, embedding_weights, X_train, y_train, X_test, y_test, X_test_l)

        test_txt = input("请输入要预测的内容")
        prodection(index_dict, test_txt)



