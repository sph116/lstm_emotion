
# lstm_emotion
## 训练步骤：
1.使用谭松波酒店评论数据集进行word2vec训练，word2vec参数为size=150，min_count=1,window=5

2.根据训练完成的词向量墨香提取词向量与索引字典，利用pickle存储为序列化数据。

3.根据索引字典提取每句特征 传入分类器进行训练，对训练过程的验证集与训练集的acc与loss进行可视化。

4.训练相关参数为 vocab_dim=100，maxlen=150，batch_size=100,n_epoch=4,input_length=150

5.最终训练效果acc=88.6% 效果较差，只做demo用。

## 注：
1.后续优化可增加word2vec的文本训练数量，本文只将情感分析语料作为词向量训练集，效果较差。

2.process_txt作为文本处理用，未开源，需要自己重写。文本至少需要简单的去停用词。
