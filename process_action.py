from TextUtils import get_text
from Deal import DealWithText
import jieba
import os

stop_words = open("C:/Users/孙佩豪/Desktop/爬虫（工作）/中国新闻网滚动页面/stop_words.txt", "r", encoding="utf-8")
stop_words = [word.replace("\n", "") for word in stop_words]

def process_txt(txt):
    """
    文本处理方法 未开源 需重写
    :param txt:
    :return:
    """
    txt = get_text(txt)  #格式化文本
    txt = DealWithText.get_pun(txt)  #符号归一化
    # wordls = jieba.lcut(txt)  #精确模式分词
    # results = [word for word in wordls if word not in stop_words] #去停用词
    return txt
