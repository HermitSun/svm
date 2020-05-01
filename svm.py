from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np
import pandas as pd

# fix random
np.random.seed(7)

# load data
# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
# call load_data with allow_pickle implicitly set to true
train_X = np.load('train/train_X.npy')
train_y = np.load('train/train_y.npy')
test_X = np.load('test/test_X.npy')
test_y = np.load('test/test_y.npy')
# restore np.load for future normal usage
np.load = np_load_old

# transformer
tf_idf_transformer = TfidfTransformer()

# fit model
count_v1 = CountVectorizer(
    stop_words='english',
    max_df=0.5,
    decode_error='ignore',
    binary=True
)
train_data = count_v1.fit_transform(train_X)
tf_idf_train = tf_idf_transformer.fit(train_data).transform(train_data)

model = LinearSVC()
model.fit(tf_idf_train, train_y)

# predict
count_v2 = CountVectorizer(
    vocabulary=count_v1.vocabulary_,
    stop_words='english',
    max_df=0.5,
    decode_error='ignore',
    binary=True
)
test_data = count_v2.fit_transform(test_X)
tf_idf_test = tf_idf_transformer.fit(test_data).transform(test_data)

# result
result = model.predict(tf_idf_test)
result_table = pd.DataFrame(
    confusion_matrix(test_y, result),
    index=['non-spam', 'spam'],
    columns=['non-spam', 'spam']
)
print(result_table)
print('precision: ', precision_score(test_y, result))
print('recall: ', recall_score(test_y, result))
