from models.visualization import smp_labelled, plot_confusion_matrix, plot_roc_curve
from models.visualization import smp_pair_both, smp_pair, all_in_one_plot
from models.helper_funcs import reverse_normalize
from models.metrics import calculate_metrics_raw, calculate_metrics_per_label
from models.metrics import METRICS, METRICS_PROB
from data_handling.data_parameters import ANTI_LABELS

import time
import numpy as np
import pandas as pd

from tabulate import tabulate
# for the moment only for the pure scikit based functions
# excluded for the moment: anns, baseline, cluster-then-predict algos
# Problems anns: wrong labels
# problems baseline: no prediction possible at the moment
# Problems cluster-then-predict: yet unclear
# later TODO split up in prediction, metrics and plotting?

# TODO make saving of plots possible
def testing(model, x_train, y_train, x_test, y_test, smp_idx_train, smp_idx_test,
            annot="test", name="Model", labels_order=None, roc_curve=False,
            confusion_matrix=False, bog_plot_preds=None, bog_plot_trues=None,
            one_plot=False, pair_plots=False, plot_list=None, only_trues=False,
            only_preds=False):
    """ Performs testing on a model. Model is fit on training data and evaluated on testing data. Prediction inclusive.
    Parameters:
        model (model): Model on which .fit and .predict can be called.
        x_train (pd.DataFrame): Training input data.
        y_train (pd.Series): Training target data - the desired output.
        x_test (pd.DataFrame): Testing input data.
        y_test (pd.Series): Testing target data.
        smp_idx_train (pd.Series): SMP indices for the tranining data.
        smp_idx_test (pd.Series): SMP indices for the testint data.
        annot (str): How the metrics results should be annotated.
        name (str): Name of the Model.
        labels_order (list): list where a wished labels order is given. If None
            the labels will be sorted automatically ascending.
        roc_curve (bool): If the roc curve should be plotted.
        confusion_matrix (bool): If the confusion matrix should be plotted.
        bog_plot_preds (str): Default None, means no bogplot is produced. If str
            the plot is produced and saved under the name given here. This one
            is only for bog plot of predicted profiles.
        bog_plot_trues (str): Default None, means no bogplot is produced. If str
            the plot is produced and saved under the name given here. This one
            is only for bog plot of true observed profiles.
        one_plot (bool): if all profiles should be displayes each in one plot.
            The observation is a bar above the plot of the predicted profile.
        pair_plots (bool): if all profiles should be displayed as prediction -
            obeservation subplot pairs.
        only_trues (bool): Only the true profiles are plotted each by each.
        only_preds (bool): Only the predicted profiles are plotted each by each.
        plot_list (list): Default None. If not None, this is a list of smp names.
            All wished plots are only produced for the named smp profiles.
            (No strings! Must be floats or numeric inside the list!)
    Returns:
        tuple: (Metrics of the results, Metrics per label and confusion matrix)
    """
    if labels_order is None:
        labels_order = np.sort(np.unique(y_test))

    # fitting the model
    start_time = time.time()
    model.fit(x_train, y_train)
    fit_time = time.time() - start_time
    # predicting
    start_time = time.time()
    y_pred = model.predict(x_test)
    score_time = time.time() - start_time
    # calculate usual metrics
    scores = calculate_metrics_raw(y_test, y_pred, metrics=METRICS, cv=False, name=name, annot=annot)
    scores[annot+"_fit_time"] = fit_time
    scores[annot+"_score_time"] = score_time

    # check if probability prediction is possible and do it if yes
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(x_test)
        prob_scores = calculate_metrics_raw(y_test, y_pred_prob, metrics=METRICS_PROB, cv=False, name=name, annot=annot)
        # merge them with the other scores
        scores = {**scores, **prob_scores}

    # calculate metrics per label and confusion matrix
    metrics_per_label = calculate_metrics_per_label(y_test, y_pred, name=name, annot=annot, labels_order=labels_order)
    # print all metrics beautifully (makes only sense on a larger scale)
    print("\nScores \n")
    print(pd.Series(scores))
    print("\nScores per label for Model {}\n".format(name))
    scores_per_label = {key:value for key, value in metrics_per_label.items() if (key != annot+"_confusion_matrix") & (key != "model")}
    scores_per_label = pd.DataFrame.from_dict(scores_per_label, orient="index", columns=[ANTI_LABELS[i] for i in labels_order])
    print(tabulate(scores_per_label, headers="keys", tablefmt="psql"))
    
    # print confusion matrix
    if confusion_matrix:
        tags = [ANTI_LABELS[label] for label in labels_order]
        plot_confusion_matrix(metrics_per_label[annot + "_" + "confusion_matrix"], labels=tags, name=name)
    # print roc auc curve
    if roc_curve and hasattr(model, "predict_proba"):
        plot_roc_curve(y_test, y_pred_prob, labels_order, name=name)

    smp_trues = []
    smp_preds = []
    if plot_list is None:
        smp_names = smp_idx_test.unique()
    else:
        smp_names = plot_list

    # determine the predicted smp profiles
    for smp_name in smp_names:
        smp = pd.DataFrame({"mean_force": x_test["mean_force"], "distance": x_test["distance"], "label": y_test, "smp_idx": smp_idx_test})
        smp = reverse_normalize(smp, "mean_force", min=0, max=45)
        smp = reverse_normalize(smp, "distance", min=0, max=1187)
        smp_wanted = smp[smp["smp_idx"] == smp_name]
        smp_trues.append(smp_wanted)

        smp_pred = smp.copy()
        smp_pred["label"] = y_pred
        smp_wanted_pred = smp_pred[smp_pred["smp_idx"] == smp_name]
        smp_preds.append(smp_wanted_pred)

    # plot all the true profiles
    if only_trues:
        for smp_name, smp_true in zip(smp_names, smp_trues):
            smp_labelled(smp_true, smp_name, title="{} SMP Profile Observed\n".format(smp_name))

    # plot all the predicted profiles
    if only_preds:
        for smp_name, smp_pred in zip(smp_names, smp_preds):
            smp_labelled(smp_pred, smp_name, title="{} SMP Profile Predicted with Model {}\n".format(smp_name, name))

    # plot all pairs of true and observed profiles
    if pair_plots:
        for smp_name, smp_true, smp_pred in zip(smp_names, smp_trues, smp_preds):
             smp_pair_both(smp_true, smp_pred, smp_name, title="Observed and with {} Predicted SMP Profile {}\n".format(name, smp_name))

    # one plot: observation as bar above everything
    if one_plot:
        for smp_name, smp_true, smp_pred in zip(smp_names, smp_trues, smp_preds):
            smp_pair(smp_true, smp_pred, smp_name, title="Observed and with {} Predicted SMP Profile {}\n".format(name, smp_name))

    # put all smps together in one plot
    if bog_plot_trues is not None:
        all_smp_trues = pd.concat(smp_trues)
        all_in_one_plot(all_smp_trues, show_indices=False, sort=True,
                        title="All Observed SMP Profiles of the Testing Data", file_name=bog_plot_trues)
    if bog_plot_preds is not None:
        all_smp_preds = pd.concat(smp_preds)
        all_in_one_plot(all_smp_preds, show_indices=False, sort=True,
                        title="All SMP Profiles Predicted with {}.".format(name), file_name=bog_plot_preds)

    # metrics must be saved from the calling function
    return (scores, metrics_per_label)
