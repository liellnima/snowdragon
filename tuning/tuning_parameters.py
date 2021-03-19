# num_cluster = [15, 30], find_num_clusters = ["acc", "sil"], with and without tsne
KMEANS_PARAMS={"num_clusters": 15 , "find_num_clusters": "acc"}

# num_components = [15, 30], cov_type = ["tied, "diag"], find_num_clusters = ["acc", "bic"], with and without tsne
GMM_PARAMS={"num_components": 15, "find_num_clusters": "acc",
            "cov_type": "tied"}

# num_components = [15, 30], cov_type = ["tied, "diag"], with and without tsne
BMM_PARAMS={"num_components": 15, "cov_type": "tied"}

# kernel = ["knn", "rbf"], alpha=[0, 0.2, 0.4]
LABEL_SPREADING_PARAMS={"kernel": "knn", "alpha": 0.2}

# no tuning!!! Only run on subset of complete data with a base model!
# there is no parser for the base_model argument, since no tuning is planned for the self trainer!
SELF_TRAINER_PARAMS={"criterion": "threshold", "base_model": None} # ATTENTION: create base_estimator for this one!

# n_estimators= [25, 100, 500], criterion=["entropy", "gini"]
# max_features=["sqrt", "log2"], max_samples=[0.4, 0.6, 0.8], resample=[False, True]
RF_PARAMS={"n_estimators": 100, "criterion": "entropy", "max_features": "sqrt",
           "max_samples": 0.6, "resample": False}

# decision_function_shape=["ovo", "ovr"], gamma=["auto", "scale"], kernel=["rbf", "sigmoid"]
SVM_PARAMS={"decision_function_shape": "ovr", "gamma": "auto", "kernel": "rbf"}

# n_neighbors=[10, 20, 50, 100, 1000]
KNN_PARAMS={"n_neighbors": 20} # weights should be always distance

# n_estimators=[10, 100, 500], sampling_strategy=["all", "not minority"]
EASY_ENSEMBLE_PARAMS={"n_estimators": 100, "sampling_strategy": "not minority"}

# rnn size=[50, 100, 150]
# batch_size=[32, 8], epochs=[100], learning_rate=[0.01, 0.001], dropout=[0, 0.2, 0.5], dense_units=[0, 100]
LSTM_PARAMS={"batch_size": 32, "epochs": 15, "learning_rate": 0.01,
             "rnn_size": 100, "dense_units": 100, "dropout": 0.2}

# rnn size=[50, 100, 150]
# batch_size=[32, 8], epochs=[100], learning_rate=[0.01, 0.001], dropout=[0, 0.2, 0.5], dense_units=[0, 100]
BLSTM_PARAMS={"batch_size": 32, "epochs": 15, "learning_rate": 0.01,
              "rnn_size": 100, "dense_units": 100, "dropout": 0.2}

# rnn size=[150]
# batch_size=[32, 8], epochs=[100], learning_rate=[0.001, 0.0001], dropout=[0, 0.5], dense_units=[0, 100],
# attention=[False, True], bidirectional=[True, False]
ENC_DEC_PARAMS={"batch_size": 32, "epochs": 15, "learning_rate": 0.001,
                "rnn_size": 100, "dense_units": 0, "dropout": 0.2,
                "attention": False, "bidirectional": False, "regularize": False}
