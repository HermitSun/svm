from os import getcwd, listdir
from os.path import isdir, isfile
from numpy import ndarray, save
from utils import resolve
from sklearn.model_selection import train_test_split


def get_email_content(dataset_dir):
    files = []
    labels = []
    # ham emails
    for f in listdir(dataset_dir):
        d = resolve(dataset_dir, f)
        if isdir(d):
            ham = resolve(d, 'ham')
            for fi in listdir(ham):
                email_path = resolve(ham, fi)
                if isfile(email_path):
                    files.append(email_path)
                    labels.append(1)
    # spam emails
    for f in listdir(dataset_dir):
        d = resolve(dataset_dir, f)
        if isdir(d):
            spam = resolve(d, 'spam')
            for fi in listdir(spam):
                email_path = resolve(spam, fi)
                if isfile(email_path):
                    files.append(email_path)
                    labels.append(0)
    # construct email matrix
    email_matrix = ndarray((len(files),), dtype=object)
    email_id = 0
    for f in files:
        with open(f, 'r', errors='ignore') as fi:
            next(fi)
            data = fi.read().replace('\n', ' ')
            email_matrix[email_id] = data
            email_id += 1
    return email_matrix, labels


# save processed data
dataset_path = resolve(getcwd(), 'dataset')
email_data, email_labels = get_email_content(dataset_path)
train_X, test_X, train_y, test_y = train_test_split(email_data, email_labels, test_size=0.3, random_state=7)
save('train/train_X.npy', train_X)
save('train/train_y.npy', train_y)
save('test/test_X.npy', test_X)
save('test/test_y.npy', test_y)
