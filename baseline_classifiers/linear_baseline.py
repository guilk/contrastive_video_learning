import numpy as np
import os
import pickle as pkl
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# X, Y = make_multilabel_classification(n_classes=5, n_labels=1,
#                                       allow_unlabeled=True,
#                                       random_state=1)

# classif = OneVsRestClassifier(SVC(kernel='linear'))
# classif.fit(X, Y)
# print(Y)

def top_n_accuracy(preds, truths, n):
    successes = 0
    best_n = np.argsort(preds, axis=1)[:,-n:]
    best_n = best_n.tolist()
    # print(best_n)
    for i in range(preds.shape[0]):
        if truths[i] in best_n[i]:
            successes += 1
    return float(successes)/preds.shape[0]


if __name__ == '__main__':

    feat_root = '/home/liangkeg/third_hand/data/YouCookII/features/multilabel_features'
    token_type = 'object'
    feat_type = '3d'
    top_n = 5

    tr_data = pkl.load(open(os.path.join(feat_root, '{}_{}_training.pkl'.format(token_type, feat_type)), 'rb'))
    ts_data = pkl.load(open(os.path.join(feat_root, '{}_{}_validation.pkl'.format(token_type, feat_type)), 'rb'))

    tr_features = normalize(tr_data['features'])
    tr_labels = tr_data['labels']
    ts_features = normalize(ts_data['features'])
    ts_labels = ts_data['labels']
    print(tr_features.shape, tr_labels.shape)

    # classifier = SVC(kernel='linear', C=100, probability=True)
    # classifier.fit(tr_features, tr_labels)
    # predictions = classifier.predict_proba(ts_features)
    # print(top_n_accuracy(predictions, ts_labels, 1))
    # predictions = classifier.predict_proba(ts_features)
    # print(top_n_accuracy(predictions, ts_labels, 5))


    # total = 0
    # for index in range(tr_labels.shape[1]):
    #     tr_label = tr_labels[:,index]
    #     if np.sum(tr_label) == 0:
    #         continue
    #     # print(np.sum(tr_label))
    #     ts_label = ts_labels[:,index]
    #     classifier.fit(tr_features, tr_label)
    #     accuracy = classifier.score(ts_features, ts_label)
    #     # majority = max(1.0 * np.sum(tr_label)/tr_features.shape[0], )
    #     total += 1.0 * np.sum(tr_label)/tr_features.shape[0]
    #     print('{},{},{}'.format(index, accuracy, 1.0 * np.sum(tr_label)/tr_features.shape[0]))
    # print(total)
    #
    classif = OneVsRestClassifier(SVC(kernel='linear', probability=True, C=100), n_jobs=8)
    classif.fit(tr_features, tr_labels)
    # mean_acc = classif.score(ts_features, ts_labels)
    predictions = classif.predict(ts_features)
    predictions = np.reshape(predictions, (predictions.shape[0]*predictions.shape[1], -1))
    ts_labels = np.reshape(ts_labels, (ts_labels.shape[0]*ts_labels.shape[1], -1))
    recall = recall_score(ts_labels, predictions, average='macro')
    precision = precision_score(ts_labels, predictions, average='macro')
    print(precision, recall)
    print(predictions.shape)
    assert False
    #
    #
    # probs = classif.predict_proba(ts_features)
    #
    # print(top_n_accuracy(probs, ts_labels, 1))
    # print(top_n_accuracy(probs, ts_labels, 5))
    #
    # # best_n = np.argsort(probs, axis=1)[:, -top_n:]
    # # print(probs)
    # # print(np.max(probs))

