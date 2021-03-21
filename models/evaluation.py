from models.visualization import smp_labelled, plot_confusion_matrix, plot_roc_curve
from models.helper_funcs import reverse_normalize
from models.metrics import calculate_metrics_raw, calculate_metrics_per_label
from models.metrics import METRICS, METRICS_PROB
from data_handling.data_parameters import ANTI_LABELS

import numpy as np
import pandas as pd


# for the moment only for the pure scikit based functions
# excluded for the moment: anns, baseline, cluster-then-predict algos
# Problems anns: wrong labels
# problems baseline: no prediction possible at the moment
# Problems cluster-then-predict: yet unclear
# TODO check if prob_predict is possible for a model
# TODO add annot for anotation purposes
def testing(model, x_train, y_train, x_test, y_test, smp_idx_train, smp_idx_test, visualization=False):
    """ Performs testing on a model. Model is fit on training data and evaluated on testing data. Prediction inclusive.
    Parameters:
        model
        x_train
        y_train
        x_test
        y_test
        smp_idx_train
        smp_idx_test
        visualization (bool): If False, only the metrics will be returned/printed.
            If True, the predictions are also visualized for all or some SMP profiles.
    """
    labels_order = np.sort(np.unique(y_test))
    annot = "test"
    name = "RandomForest"

    # TODO add score and fit time
    # fitting the model
    model.fit(x_train, y_train)
    # predicting and calculating metrics
    y_pred = model.predict(x_test)
    scores = calculate_metrics_raw(y_test, y_pred, metrics=METRICS, cv=False, name=name, annot=annot)

    # check if probability prediction is possible and do it if yes
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(x_test)
        prob_scores = calculate_metrics_raw(y_test, y_pred_prob, metrics=METRICS_PROB, cv=False, name=name, annot=annot)

    # calculate metrics per label and confusion matrix
    metrics_per_label = calculate_metrics_per_label(y_test, y_pred, name=name, annot=annot, labels_order=labels_order)
    print("Scores per label\n", metrics_per_label)
    print("Scores \n", scores)
    print("Probability Scores \n", prob_scores)
    tags = [ANTI_LABELS[label] for label in labels_order]
    #plot_confusion_matrix(metrics_per_label[annot + "_" + "confusion_matrix"], labels=tags, name=name)
    print("hello!")
    plot_roc_curve(y_test, y_pred_prob, labels_order, name=name)

    exit(0)

    # visualization for each smp profile
    for smp_name in smp_idx_test.unique():
        smp = pd.DataFrame({"mean_force": x_test["mean_force"], "distance": x_test["distance"], "label": y_test, "smp_idx": smp_idx_test})
        smp = reverse_normalize(smp, "mean_force", min=0, max=45)
        smp = reverse_normalize(smp, "distance", min=0, max=1187)
        smp.info()
        smp_wanted = smp[smp["smp_idx"] == smp_name]

        smp_labelled(smp_wanted, smp_name)
        smp_pred = smp.copy()
        smp_pred["label"] = y_pred
        smp_wanted_pred = smp_pred[smp_pred["smp_idx"] == smp_name]
        smp_labelled(smp_wanted_pred, smp_name)

    exit(0)


    # pick out a certain smp profile in the test set:
