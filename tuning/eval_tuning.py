from tuning.tuning_parameters import BEST_PARAMS

import pandas as pd

from tabulate import tabulate

# longterm TODO add parser, such that the results of a given csv is evaluated
# longterm TODO try random forest with 100 or even 15 -> random forest has been overfitting already

def read_tuning_results():
    """ Reads in a csv and finds the best hyperparameters in terms of balanced
    accuracy. Saves the results in pretty print tables. The tables contain only
    relevant hyperparameters.
    """
    results_csv = "tuning/tuning_results/tuning_run01_server.csv"
    results = pd.read_csv(results_csv)

    # produce group results: fit_time, score_time, acc, prec, recall, roc_auc, log loss
    # important in case there is no hyperparmeter tuning!
    print("Mean Crossvalidation scores of each model type:\n")
    mean_results = results.groupby(["model"]).mean()[["fit_time", "score_time", "test_roc_auc", "test_log_loss", "test_precision", "test_recall", "test_balanced_accuracy"]]
    mean_results_sorted = mean_results.sort_values(["test_recall"], ascending=False)
    print(tabulate(mean_results_sorted, headers="keys", tablefmt="psql"))
    # save as latex table!
    with open("tuning/tuning_results/tables/models_mean_metrics.txt", 'w') as f:
        f.write(tabulate(mean_results_sorted, headers="keys", tablefmt="latex_raw"))

    # get the relevant params for each model
    params = {model: list(params_dict.keys()) for model, params_dict in BEST_PARAMS.items()}
    print(params)
    max_accs = results.groupby("model")["test_recall"].max()
    all_best_metrics = []
    all_best_params = []

    # read out max results and respective metrics for each type of model
    for model in results["model"].unique():
        # get only first best model
        best_model = results[(results["model"] == model) & (results["test_recall"] == max_accs.loc[model])].iloc[0]
        best_metrics = best_model[["model", "fit_time", "score_time", "test_roc_auc", "test_log_loss", "test_precision", "test_recall", "test_balanced_accuracy"]]
        best_params = best_model[params[model]]
        all_best_metrics.append(best_metrics)
        all_best_params.append(best_params)
        # print best params
        print("\nBest Hyperparameters for Model {}:".format(model))
        print(best_params)

    max_results = pd.concat(all_best_metrics, axis=1).transpose().sort_values(["test_balanced_accuracy"], ascending=False)
    print("\nBest Crossvalidation scores of each model type:\n")
    print(tabulate(max_results, headers="keys", showindex=False, tablefmt="psql"))
    with open("tuning/tuning_results/tables/models_max_metrics.txt", 'w') as f:
        f.write(tabulate(max_results, headers="keys", showindex=False, tablefmt="latex_raw"))

    # what we want: a table with all models of one type, but only with the relevant parameters
    for model in results["model"].unique():
        model_results = results[results["model"] == model]
        # filter for wanted cols and sort
        wanted_cols = params[model] + ["fit_time", "score_time", "test_roc_auc", "test_log_loss", "test_precision", "test_recall", "test_balanced_accuracy"]
        model_results = model_results[wanted_cols]
        model_results = model_results.sort_values(["test_balanced_accuracy"], ascending=False)
        # rename column test_balanced_accuracy
        model_results = model_results.rename(columns={"test_balanced_accuracy": "test_bal_acc", "test_precision": "test_prec"})
        print("\nComplete Results for Model {}".format(model))
        print(tabulate(model_results, headers="keys", showindex=False, tablefmt="psql"))
        with open("tuning/tuning_results/tables/results_{}.txt".format(model), 'w') as f:
            f.write(tabulate(model_results, headers="keys", showindex=False, tablefmt="latex_raw"))

if __name__ == "__main__":
    read_tuning_results()
