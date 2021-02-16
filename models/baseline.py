
import time
import numpy as np

# TODO docu
def majority_class_baseline(x_train, y_train, cv):
    y_preds_train = []
    y_trues_train = []
    y_preds_valid = []
    y_trues_valid = []
    all_fit_time = []
    all_score_time = []

    for k in cv:
        # current target values for this fold (training and validation)
        fit_time = time.time()
        fold_y_train = y_train[k[1]]
        all_fit_time.append(time.time() - fit_time)
        fold_y_valid = y_train[k[0]]
        # append true labels for current fold (both for training and validation data)
        y_trues_train.append(fold_y_train)
        y_trues_valid.append(fold_y_valid)
        # majority class in this fold of training data (will be also the majority class for the validation set!)
        score_time = time.time()
        maj_class = fold_y_train.mode()
        y_pred = pd.Series(np.repeat(maj_class, len(fold_y_valid))) # predicted labels of validation data
        all_score_time.append(time.time() - score_time)

        y_preds_valid.append(y_pred)
        y_preds_train.append(pd.Series(np.repeat(maj_class, len(fold_y_train)))) # predicted labels of training data


    train_scores = calculate_metrics_raw(y_trues_train, y_preds_train, name="MajorityClassBaseline", annot="train")
    test_scores = calculate_metrics_raw(y_trues_valid, y_preds_valid, name="MajorityClassBaseline", annot="test")

    scores = {**train_scores, **test_scores}
    scores["fit_time"] = np.asarray(all_fit_time)
    scores["score_time"] = np.asarray(all_score_time)

    return scores
