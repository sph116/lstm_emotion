# from TextUtils import get_text
# from Deal import DealWithText
import jieba
import os

stop_words = open("./stop_words.txt", "r", encoding="utf-8")
stop_words = [word.replace("\n", "") for word in stop_words]

def del_stop_words(text):
	"""
	删除每个文本中的停用词
	:param text:
	:return:
	"""
	word_ls = jieba.lcut(text)
	word_ls = [i for i in word_ls if i not in stopwords]
	return word_ls
