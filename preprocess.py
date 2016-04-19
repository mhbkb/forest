__author__ = 'maohao'
from nltk.stem.snowball import SnowballStemmer
import loaddata
import numpy as np
import re
import string


def removePunctuation(text):
    result = ''
    for s in re.sub('[%s]' % re.escape(string.punctuation), '', text).strip().split():
        result += s.lower() + ' '
    return result.strip()

# print(removePunctuation('Hi, you! &'))
# print(removePunctuation(' No under_score!'))
# print(removePunctuation(" The Elephant's 4 cats. "))


def digitize(text):
    text = text.lower()
    strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
    return (" ").join([str(strNum[z]) if z in strNum else z for z in text.split(" ")])

# print(digitize(' one cat'))
# print(digitize('two honeybee'))
# print(digitize(" Three Elephants"))


def unitize(s):
    s = re.sub(r"([0-9]+)( *)(inches|inch|in|'')\.?", r"\1in. ", s)
    s = re.sub(r"([0-9]+)( *)(foot|feet|ft|')\.?", r"\1ft. ", s)
    s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
    s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(gallons|gallon|gals|gal)\.?", r"\1gal. ", s)
    s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
    s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
    s = re.sub(r"([0-9]+)( *)(millimeters|milimeters|mm)\.?", r"\1mm. ", s)
    s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
    s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
    s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
    s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
    return s

# print(unitize(' 5 inches'))
# print(unitize(' 342  ounces'))
# print(unitize(" 880degrees"))


def stem(s):
    s = removePunctuation(s)
    s = digitize(s)
    s = unitize(s)
    stemmer = SnowballStemmer('english')
    s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
    s = s.replace("  "," ")
    return s

# print(stem(' 5 inches  '))
# print(stem('   342  ounces  '))
# print(stem(" 880degrees  "))


def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]"," ", str2)
    str2 = [z for z in set(str2.split()) if len(z)>2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word)>3:
            s1 = []
            s1 += segmentit(word,str2,True)
            if len(s)>1:
                s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))


def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return r


def prepareData():
    #Load data
    all_data = loaddata.load_data()
    data = all_data[1]

    #Stem attributes
    data['search_term'] = data['search_term'].map(lambda x:stem(x))
    data['product_title'] = data['product_title'].map(lambda x:stem(x))
    data['product_description'] = data['product_description'].map(lambda x:stem(x))
    print('start brand info!')
    data['brand'] = data['brand'].map(lambda x:stem(x))
    print('finish brand info!')
    data['bullet1'] = data['bullet1'].map(lambda x:stem(x))
    data['bullet2'] = data['bullet2'].map(lambda x:stem(x))
    data['bullet3'] = data['bullet3'].map(lambda x:stem(x))
    data['bullet4'] = data['bullet4'].map(lambda x:stem(x))
    data['material'] = data['material'].map(lambda x:stem(x))

    data['product_info'] = data['search_term']+"\t"+data['product_title'] +"\t"+data['product_description']

    # Calculate length
    data['len_of_query'] = data['search_term'].map(lambda x:len(x.split())).astype(np.int64)
    data['len_of_title'] = data['product_title'].map(lambda x:len(x.split())).astype(np.int64)
    data['len_of_description'] = data['product_description'].map(lambda x:len(x.split())).astype(np.int64)
    data['len_of_brand'] = data['brand'].map(lambda x:len(x.split())).astype(np.int64)
    data['len_of_b1'] = data['bullet1'].map(lambda x:len(x.split())).astype(np.int64)
    data['len_of_b2'] = data['bullet2'].map(lambda x:len(x.split())).astype(np.int64)
    data['len_of_b3'] = data['bullet3'].map(lambda x:len(x.split())).astype(np.int64)
    data['len_of_b4'] = data['bullet4'].map(lambda x:len(x.split())).astype(np.int64)

    # Search and Query
    data['search_term'] = data['product_info'].map(lambda x:seg_words(x.split('\t')[0],x.split('\t')[1]))
    data['attr'] = data['search_term']+"\t"+data['brand']
    data['bullets'] = data['search_term']+"\t"+data['bullet1']+"\t"+data['bullet2']+"\t"+data['bullet3']+"\t"+data['bullet4']

    data.to_csv('features.csv', sep='\t', encoding='ISO-8859-1')
    return all_data

prepareData()