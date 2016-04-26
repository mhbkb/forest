import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

stem = SnowballStemmer('english')

data_train = pd.read_csv('files/train.csv', encoding="ISO-8859-1")
data_test = pd.read_csv('files/test.csv', encoding="ISO-8859-1")
data_desc = pd.read_csv('files/product_descriptions.csv')

train_count = data_train.shape[0]


def str_stemmer(s):
	return " ".join([stem.stem(word) for word in s.lower().split()])


def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())



data_all = pd.concat((data_train, data_test), axis=0, ignore_index=True)

data_all = pd.merge(data_all, data_desc, how='left', on='product_uid')

data_all['search_term'] = data_all['search_term'].map(lambda x:str_stemmer(x))
data_all['product_title'] = data_all['product_title'].map(lambda x:str_stemmer(x))
data_all['product_description'] = data_all['product_description'].map(lambda x:str_stemmer(x))

data_all['len_of_query'] = data_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

data_all['product_info'] = data_all['search_term']+"\t"+data_all['product_title']+"\t"+data_all['product_description']

data_all['word_in_title'] = data_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
data_all['word_in_description'] = data_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

data_all = data_all.drop(['search_term','product_title','product_description','product_info'],axis=1)

data_train = data_all.iloc[:train_count]
data_test = data_all.iloc[train_count:]
id_test = data_test['id']

y_train = data_train['relevance'].values
X_train = data_train.drop(['id','relevance'],axis=1).values
X_test = data_test.drop(['id','relevance'],axis=1).values

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission3.csv',index=False)