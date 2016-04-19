import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import pipeline, grid_search
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer

TRAIN_COUNT = 74067

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title',
            'product_description','product_info','attr','brand','bullets',
            'bullet1','bullet2','bullet3','bullet4', 'len_of_b1',
            'len_of_b2', 'len_of_b3', 'len_of_b4', 'material']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE = make_scorer(fmean_squared_error, greater_is_better=False)

def process():
    train_count = TRAIN_COUNT
    data = pd.read_csv('features.csv', encoding="ISO-8859-1", sep='\t')
    print('load successfully!')

    train = data.iloc[:train_count]
    test = data.iloc[train_count:]
    test_id = test['id']
    train_y = train['relevance'].values
    train_x = train[:]
    test_x = test[:]

    print('randomForest starts:')
    # n_estimators how many trees in forest, default 10
    rfr = RandomForestRegressor(n_estimators = 500, random_state = 31, verbose = 1, n_jobs = -1)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    tsvd = TruncatedSVD(n_components=15, random_state = 31337)
    clf = pipeline.Pipeline([
            ('union', FeatureUnion(
                        transformer_list = [
                            ('cst',  cust_regression_vals()),
                            ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                            ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                            ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
                            ]
                    )),
            ('rfr', rfr)])
    param_grid = {
                    'rfr__n_estimators': [123,125,127],
                    'rfr__max_depth': [24],
                    'rfr__max_features': [18]
                 }
    model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid,
        cv = 10, verbose = 250, scoring=RMSE)
    model.fit(train_x, train_y)

    print("Best parameters found by grid search:")
    print(model.best_estimator_)
    print("Best CV score:")
    print(model.best_score_)

    y_pred = model.predict(test_x)
    print(len(y_pred))
    pd.DataFrame({"id": test_id, "relevance": y_pred}).to_csv('submisson.csv',index=False)


process()