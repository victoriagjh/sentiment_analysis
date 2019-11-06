import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
from nltk import FreqDist

def annotation_graph(wordcounter, pos_wordcount, neg_wordcount):
    pos = (pos_wordcount/wordcounter) * 100
    neg = (neg_wordcount/wordcounter) * 100
    neu = ((wordcounter-pos_wordcount-neg_wordcount)/wordcounter)*100

    A = [pos]
    B = [neu] 
    C = [neg] 

    #회전 전
    plt.rcParams["figure.TOfigsize"] = (2, 15)
    c_bottom = np.add(A, B)
    plt.bar(1, A, color='red')
    plt.bar(1, B, bottom = A, color='orange')
    plt.bar(1, C, bottom = c_bottom, color='#92D050')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.savefig('fig1.png', dpi=300)

    #회전 후 
    plt.figure(figsize=(12, 3))
    img = mping.imread('fig1.png').transpose(1,0,2)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('fig1.png', dpi=300)
    plt.show()


#frequent 함수 positiveTopFrequentWords, negativeTopFrequentWords
#[('Mr.', 1), ('Mrs.', 1), ('Miss', 1), ('Mr', 1), ('Mrs', 1)]
def frequent_graph(frequent_list):
    fd_content = FreqDist(list)
    frequent = fd_content.most_common(5)
    X = []
    Y = []
    for i in range(5):
        X.append(frequent[i][0])
        Y.append(frequent[i][1])

    plt.bar(X, Y)
    plt.savefig('frequent_word.png')

