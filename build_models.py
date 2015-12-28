import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib


def build_model():
    print "Loading data..."
    df = pd.read_pickle('song_comps_added_features.pkl')
    X = df.drop('genre', axis=1).values
    Y = df.genre.values
    # shuffle the indices bc data is sorted by genre
    ind = np.random.permutation(len(Y))

    n_samp = 40000
    train_X = X[ind[:n_samp], :]
    train_Y = Y[ind[:n_samp]]

    # save the test set for model evaluation
    test_X = X[ind[75000:], :]
    np.save('test_X', test_X)
    test_Y = Y[ind[75000:]]
    np.save('test_Y', test_Y)

    print "Building SVM with rbf kernel..."

    rbf_svm = SVC(C=10, kernel='rbf', probability=True)
    rbf_svm.fit(train_X, train_Y)
    joblib.dump(rbf_svm, 'rbf_svm.pkl')


def main():
    build_model()

if __name__ == '__main__':
    main()
