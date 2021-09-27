from models.metrics import METRICS, METRICS_PROB
from models.helper_funcs import reverse_normalize, int_to_idx, save_results
from data_handling.data_parameters import ANTI_LABELS
from models.semisupervised_models import assign_clusters
from models.anns import fit_ann_model, predict_ann_model
from models.baseline import fit_baseline, predict_baseline
from models.metrics import calculate_metrics_raw, calculate_metrics_per_label
from visualization.plot_data import all_in_one_plot, plot_confusion_matrix, plot_roc_curve
from visualization.plot_profile import smp_pair_both, smp_pair, smp_labelled

import time
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from tabulate import tabulate
from sklearn.svm import SVC

# Longterm TODO: make more of the parameters optional!

def predicting(model, x_train, y_train, x_test, y_test, smp_idx_train, smp_idx_test, unlabelled_data, impl_type, **fit_params):
    """ Used from evaluation to predict the labels and label probabilities for
    all the different models.
    Parameters:
        model (model): Model on which .fit and .predict can be called. (or baseline )
        x_train (pd.DataFrame): Training input data.
        y_train (pd.Series): Training target data - the desired output.
        x_test (pd.DataFrame): Testing input data.
        y_test (pd.Series): Testing target data.
        smp_idx_train (pd.Series): SMP indices for the tranining data.
        smp_idx_test (pd.Series): SMP indices for the testint data.
        unlabelled_data (pd.DataFrame): Unlabelled Data used for semi-supervised
            learning.
        impl_type (str): Default \"scikit\". Indicates which type of model must be
            used during prediction and fitting. The following types exists:
            \"scikit\" (most models), \"baseline\", \"keras\" (for all ann models),
            \"semi_manual\" (kmeans, gmm, bmm)
        **fit_params: These are additional parameters which are given to the
            fit call. This is important e.g. for anns, since epochs and
            batch size must be be specified during fitting.
    Returns:
        (np.ndarray, np.ndarray, float, float): y_pred, y_pred_prob, fit_time, score_time.
            tuple of two np.ndarrays (length_of_xtest,) (length_of_xtest, num_of_labels)
            and two floats
    """
    # set y_pred_prob to None in case the probabilities can't be predicted
    y_pred_prob = None

    # fitting and prediction based on the model implementation type
    if impl_type == "scikit":
        # fitting the model
        start_time = time.time()
        model.fit(x_train, y_train)
        fit_time = time.time() - start_time
        # predicting
        start_time = time.time()
        y_pred = model.predict(x_test)
        score_time = time.time() - start_time
        # predict proba - special case SVM: change model, fit newly and predict
        if isinstance(model, SVC):
            # change model
            model.probability = True
            model.fit(x_train, y_train)
        # predicting probability
        if hasattr(model, "predict_proba"): y_pred_prob = model.predict_proba(x_test)

    elif impl_type == "keras":
        # fitting the model
        start_time = time.time()
        fit_ann_model(model, x_train, y_train, smp_idx_train, **fit_params)
        fit_time = time.time() - start_time
        # predicting + predicting probability
        start_time = time.time()
        y_pred, y_pred_prob = predict_ann_model(model, x_test, y_test, smp_idx_test, predict_proba=True, **fit_params)
        score_time = time.time() - start_time

    elif impl_type == "baseline":
        # fitting the model
        start_time = time.time()
        majority_vote = fit_baseline(y_train)
        fit_time = time.time() - start_time
        # predicting + predicting probability
        start_time = time.time()
        y_pred = predict_baseline(majority_vote, x_test)
        score_time = time.time() - start_time

    elif impl_type == "semi_manual":
        # fitting the model
        start_time = time.time()
        # concat unlabelled and labelled data and fit it
        model = model.fit(pd.concat([unlabelled_data, x_train]))
        fit_time = time.time() - start_time

        # determine the number of clusters/components
        if hasattr(model, "cluster_centers_"):
            num_components = model.cluster_centers_.shape[0]
        elif hasattr(model, "weights_"):
            num_components = model.weights_.shape[0]
        # prediction
        start_time = time.time()
        test_clusters = model.predict(x_test)
        y_pred = assign_clusters(y_test, test_clusters, num_components)
        score_time = time.time() - start_time

    else:
        raise ValueError("""This Model implementation types does not exist.
        Choose one of the following: \"scikit\" (for rf, svm, knn, easy_ensemble, self_trainer),
        \"semi_manual\" (for kmean, gmm, bmm), \"keras\" (for lstm, blsm, enc_dec),
        or \"baseline\" (for the majority vote baseline)""")

    return y_pred, y_pred_prob, fit_time, score_time

def metrics_testing(y_test, y_pred, y_pred_prob, fit_time, score_time, labels_order, annot="test", name="Model", save_dir=None, printing=False):
    """ Calculates and prints metrics.
    Parameters:
        y_test (pd.Series): Testing target data.
        y_pred (np.array): Predictions of the y_test data.
        y_pred_prob (np.ndarray): 2dim array with probability predictions of y_test.
            Can be also None.
        fit_time (float): fitting time
        score_time (float): scoring time
        labels_order (list): list where a wished labels order is given.
        save_dir (str): Default None means metrics are printed but not saved. If
            str the metrics are saved in this folder but they are not printed.
        annot (str): How the metrics results should be annotated.
        name (str): Name of the Model.
        printing (bool): If results should be printed
    Returns:
        (dict, dict): tuple of dictionaries. The first one contains the general metrics,
            the second one contains the metrics calculated per metric.
    """
    # calculate usual metrics
    scores = calculate_metrics_raw(y_test, y_pred, metrics=METRICS, cv=False, name=name, annot=annot)
    scores[annot+"_fit_time"] = fit_time
    scores[annot+"_score_time"] = score_time

    # check if probability pred scores exist, if yes add those scores
    if y_pred_prob is not None:
        prob_scores = calculate_metrics_raw(y_test, y_pred_prob, metrics=METRICS_PROB, cv=False, name=name, annot=annot)
        scores = {**scores, **prob_scores}

    # calculate metrics per label and confusion matrix
    metrics_per_label = calculate_metrics_per_label(y_test, y_pred, name=name, annot=annot, labels_order=labels_order)
    scores_per_label = {key:value for key, value in metrics_per_label.items() if (key != annot+"_confusion_matrix") & (key != "model")}
    scores_per_label = pd.DataFrame.from_dict(scores_per_label, orient="index", columns=[ANTI_LABELS[i] for i in labels_order])

    # print all metrics beautifully (makes only sense on a larger scale)
    if printing:
        print("\nScores \n")
        print(pd.Series(scores))
        print("\nScores per label for Model {}\n".format(name))
        print(tabulate(scores_per_label, headers="keys", tablefmt="psql"))

    # for returning later
    scores_output = pd.DataFrame(columns=scores.keys())
    scores_output.loc[0] = scores.values()

    if save_dir is not None:
        # for saving
        del scores["model"]
        scores= pd.DataFrame({"Metrics": scores.keys(), "Values": scores.values(), "Model": [name] * len(scores.keys())})
        # save scores
        scores.to_csv(save_dir + "/scores.csv")
        # save scores per label
        scores_per_label.to_csv(save_dir + "/scores_per_label.csv")

    return scores_output, metrics_per_label

def plot_testing(y_pred, y_pred_prob, metrics_per_label, x_test, y_test,
                 smp_idx_test, labels_order, annot="test", name="Model",
                 roc_curve=False, confusion_matrix=False, bog_plot_preds=None,
                 bog_plot_trues=None, one_plot=False, pair_plots=False,
                 only_trues=False, only_preds=False, plot_list=None,
                 save_dir=None, **kwargs):
    """ Plots visualization for predictions and metrics.
    Parameters:
        y_pred (np.array): all predictions of the x_test
        y_pred_prob (np.ndarray): contains probability predictions for x_test
        metrics_per_label (dict): contains metrics, such as the confusion matrix
        x_test (pd.DataFrame): Testing input data.
        y_test (pd.Series): Testing target data.
        smp_idx_test (pd.Series): SMP indices for the testing data.
        labels_order (list): list where a wished labels order is given.
        annot (str): How the metrics results should be annotated.
        name (str): Name of the Model.
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
            If None, all SMP profiles are plotted.
        save_dir (str): Default None means that the plots are not saved.
            If this is a string, the plots will be saved in this folder.
    """
    # print confusion matrix
    if confusion_matrix:
        print("\tPlotting Confusion Matrix...")
        tags = [ANTI_LABELS[label] for label in labels_order]
        plot_confusion_matrix(metrics_per_label[annot + "_" + "confusion_matrix"], labels=tags, name=name, save_file=save_dir + "/confusion_matrix.png")
        print("\t...done.\n")

    # print roc auc curve
    if roc_curve and (y_pred_prob is not None):
        print("\tPlotting ROC Curve...")
        plot_roc_curve(y_test, y_pred_prob, labels_order, name=name, save_file=save_dir + "/roc_curve.png")
        print("\t...done.\n")

    smp_trues = []
    smp_preds = []
    if plot_list is None:
        smp_names = smp_idx_test.unique()
    else:
        smp_names = plot_list


    # determine the predicted smp profiles
    print("\tCalculating smp predictions:")
    for smp_name in tqdm(smp_names):
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
        print("\tPlotting all true smp profiles:")
        for smp_name, smp_true in tqdm(zip(smp_names, smp_trues), total=len(smp_names)):
            smp_name_str = int_to_idx(smp_name)
            save_file = save_dir + "/trues/smp_" + smp_name_str + ".png" if save_dir is not None else None
            smp_labelled(smp_true, smp_name, title="{} SMP Profile Observed\n".format(smp_name_str), save_file=save_file)

    # plot all the predicted profiles
    if only_preds:
        print("\tPlotting all predicted smp profiles:")
        for smp_name, smp_pred in tqdm(zip(smp_names, smp_preds), total=len(smp_names)):
            smp_name_str = int_to_idx(smp_name)
            save_file = save_dir + "/preds/smp_" + smp_name_str + ".png" if save_dir is not None else None
            smp_labelled(smp_pred, smp_name, title="SMP Profile {}\nPredicted with {}".format(smp_name_str, name), save_file=save_file)

    # plot all pairs of true and observed profiles
    if pair_plots:
        print("\tPlotting all pair plots:")
        for smp_name, smp_true, smp_pred in tqdm(zip(smp_names, smp_trues, smp_preds), total=len(smp_names)):
            smp_name_str = int_to_idx(smp_name)
            save_file = save_dir + "/pairs/smp_" + smp_name_str + ".png" if save_dir is not None else None
            smp_pair_both(smp_true, smp_pred, smp_name, title="Observed and with {} Predicted\nSMP Profile {}".format(name, smp_name_str), save_file=save_file)

    # one plot: observation as bar above everything
    if one_plot:
        print("\tPlotting all onesies:")
        for smp_name, smp_true, smp_pred in tqdm(zip(smp_names, smp_trues, smp_preds), total=len(smp_names)):
            smp_name_str = int_to_idx(smp_name)
            save_file = save_dir + "/onesies/smp_" + smp_name_str + ".png" if save_dir is not None else None
            smp_pair(smp_true, smp_pred, smp_name, title="Observed and with {} Predicted\nSMP Profile {}".format(name, smp_name_str), save_file=save_file)

    # put all smps together in one plot
    if bog_plot_trues is not None:
        print("\tPlotting true bogplots:")
        all_smp_trues = pd.concat(smp_trues)
        save_file = save_dir + "/bogplot_trues.png" if save_dir is not None else None
        all_in_one_plot(all_smp_trues, show_indices=False, sort=True,
                        title="All Observed SMP Profiles of the Testing Data", file_name=save_file)

    if bog_plot_preds is not None:
        print("\tPlotting predicted bogplots:")
        all_smp_preds = pd.concat(smp_preds)
        save_file = save_dir + "/bogplot_preds.png" if save_dir is not None else None
        all_in_one_plot(all_smp_preds, show_indices=False, sort=True,
                        title="All SMP Profiles Predicted with {}".format(name), file_name=save_file)

# TODO during testing: write out results for all models into csv to make united
# results plots possible (delete single plots later)
def testing(model, x_train, y_train, x_test, y_test, smp_idx_train, smp_idx_test,
            unlabelled_data=None, smoothing=0, annot="test", name="Model", labels_order=None,
            impl_type="scikit", save_dir=None, save_visualization_data=False, printing=False, **plot_and_fit_params):
    """ Performs testing on a model. Model is fit on training data and evaluated on testing data. Prediction inclusive.
    Parameters:
        model (model): Model on which .fit and .predict can be called. (or baseline )
        x_train (pd.DataFrame): Training input data.
        y_train (pd.Series): Training target data - the desired output.
        x_test (pd.DataFrame): Testing input data.
        y_test (pd.Series): Testing target data.
        smp_idx_train (pd.Series): SMP indices for the tranining data.
        smp_idx_test (pd.Series): SMP indices for the testing data.
        unlabelled_data (pd.DataFrame): Unlabelled Data used for semi-supervised
            learning.
        smoothing (int): Use this parameter when the predicted results should be
            smoothed. Default 0 means no smoothing (1 is also no smoothing).
            The value represents the size of the smoothing window.
        annot (str): How the metrics results should be annotated.
        name (str): Name of the Model.
        labels_order (list): list where a wished labels order is given. If None
            the labels will be sorted automatically ascending. If ANN models with
            predicting probabilities are used, the order must be ascending.
        impl_type (str): Default \"scikit\". Indicates which type of model must be
            used during prediction and fitting. The following types exists:
            \"scikit\" (most models), \"baseline\", \"keras\" (for all ann models),
            \"semi_manual\" (kmeans, gmm, bmm)
        save_dir (str): Default None, means that the plots and metrics are shown,
            but not saved. If str, the plots and metrics are saved there without
            showing.
        save_visualization_data (bool): Default True. Indicates if the raw data
            should be saved for later visualization processes. The data is saved
            in the subdir "eval_raw_data" of the model folder, hence save_dir
            must not be "None".
        printing (bool): if metrics for each model should be printed. Default False.
        **plot_and_fit_params: contains:
            **plotting: contains booleans about plotting. And a param for saving
                the plots and a list to indicate the wished smp profiles for
                which plotting should be done. See plot_testing.
            **fit_params: These are additional parameters which are given to the
                fit call. This is important e.g. for anns, since epochs and
                batch size must be be specified during fitting.
    Returns:
        tuple: (Metrics of the results and Metrics per label)
    """
    if labels_order is None:
        labels_order = np.sort(np.unique(y_test))

    # make predictions for models. y_pred_prob is None if not possible
    print("\tCalculating Predictions...")
    y_pred, y_pred_prob, fit_time, score_time = predicting(model, x_train,
                y_train, x_test, y_test, smp_idx_train, smp_idx_test,
                unlabelled_data, impl_type, **plot_and_fit_params)
    print("\t...done.\n")

    ############## SMOOTHING ###################################################
    # use a majority vote inside the window
    if smoothing > 1:
        center_pos = int(smoothing / 2 - 0.5 if smoothing % 2 else smoothing / 2)
        y_pred_old = y_pred # save to replace resulting nans
        y_pred_new = np.array(pd.Series(y_pred).rolling(smoothing, center=True).apply(lambda x: np.array(x)[center_pos] if len(x.mode()) > 1 else x.mode()[0]))
        # replace nans
        y_pred = np.array([old_val if np.isnan(new_val) else new_val for new_val, old_val in zip(y_pred_new, y_pred_old)])
    ############################################################################

    # create dirs for metrics and plots
    if save_dir is not None:
        save_dir_path = Path.cwd() / save_dir
        # check if dir exists and create it if not
        if not save_dir_path.is_dir():
            save_dir_path.mkdir(parents=True, exist_ok=True)
            # create subdirs as well
            Path(save_dir_path / "trues").mkdir(parents=True, exist_ok=True)
            Path(save_dir_path / "preds").mkdir(parents=True, exist_ok=True)
            Path(save_dir_path / "pairs").mkdir(parents=True, exist_ok=True)
            Path(save_dir_path / "onesies").mkdir(parents=True, exist_ok=True)
            Path(save_dir_path / "eval_raw_data").mkdir(parents=True, exist_ok=True)

    # print metrics
    print("\tCalculating Metrics...")
    scores, metrics_per_label = metrics_testing(y_test, y_pred, y_pred_prob,
                fit_time, score_time, labels_order, annot=annot, name=name, save_dir=save_dir, printing=printing)
    print("\t...done.\n")

    # plot everything
    plot_testing(y_pred, y_pred_prob, metrics_per_label, x_test, y_test,
                smp_idx_test, labels_order, annot=annot, name=name, save_dir=save_dir, **plot_and_fit_params)
    # some plots can only be done after testing of all models, in this case save raw data here
    if save_visualization_data:
        if save_dir is None:
            raise ValueError("save_dir cannot be None, if you want to save raw data for later visualization (i.e. save_visualization_data is currently set to True).")
        # save the relevant data somewhere: metrics_per_label, labels_order, y_test, y_pred_probs
        raw_data_path = str(save_dir_path) + "/eval_raw_data/"
        save_results(raw_data_path + "metrics_per_label.pickle", metrics_per_label)
        save_results(raw_data_path + "labels_order.pickle", labels_order)
        save_results(raw_data_path + "y_test.pickle", y_test)
        save_results(raw_data_path + "y_pred_prob.pickle", y_pred_prob)


    # metrics must be saved from the calling function
    return (scores, metrics_per_label)
