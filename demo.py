from snownlp import SnowNLP
import io
import os
def sentiment(path):
    f = io.open(path, encoding='utf-8')
    list = f.readline()
    sentimentlist = []
    for i in list:
        s = SnowNLP(i)
        print('s:', s.sentiments)


if __name__ == '__main__':
    sentiment('/media/omnisky/ubuntu/zxz/测试.txt')