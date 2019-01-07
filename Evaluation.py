import numpy as np
from sklearn.metrics import f1_score
import scipy.optimize as opt
from Dataset import LABEL_MAP
def F1_score_batch(pred, true, threshold=0.5):
    f1 = f1_score(pred>threshold, true, average='macro')
    return(f1)

def do_acc(preds,targs,th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()

def best_threshold_score1( pred, true):
    params = 0.5 * np.ones(28)
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(pred, true, p) - 1.0,
                                      wd * (p - 0.5)), axis=None)
    th, success = opt.leastsq(error, params)
    th[th < 0.1] = 0.1
    score = f1_score(true, pred > th, average='macro')
    return score, th

def F1_soft(preds, targs, th=0.5, d=50.0):
    preds = sigmoid_np(d * (preds - th))
    targs = targs.astype(np.float64)
    score = 2.0 * (preds * targs).sum(axis=0) / ((preds + targs).sum(axis=0) + 1e-6)
    return score

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def f1_np(y_true, y_pred):
    eps = 1e-8
    y_pred = np.round(y_pred)
    # This sums over all examples
    tp = np.sum((y_true * y_pred).astype('float'), axis=0)
    fp = np.sum(((1 - y_true) * y_pred).astype('float'), axis=0)
    fn = np.sum((y_true * (1 - y_pred)).astype('float'), axis=0)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = 2 * p * r / (p + r + eps)
    f1 = np.where(np.isnan(f1), np.zeros_like(f1), f1)
    return np.mean(f1)

def best_threshold_score2(pred, true):
    threshold = np.arange(0.1, 1.0, 0.05)
    best_list = []
    class_th = []
    for i in range(28):
        score_list = []
        for th in threshold:
            x = pred[:, i]  # 解决x和output[:,i]共用相同的存储
            p = x + 0.0
            p[p > th] = 1
            p[p <= th] = 0
            t = true[:, i]
            # score = f1_np(t, p)
            score = f1_score(t, p, average='binary')
            score_list.append(score)
        best_score = max(score_list)
        max_index = np.argmax(np.asarray(score_list))
        log = "{} best score = {:.4f} th = {}".format(LABEL_MAP[i], best_score, threshold[max_index])
        print(log)
        best_list.append(best_score)
        class_th.append(threshold[max_index])
    final_score = np.array(best_list).mean()
    return final_score, class_th
