# First argument $1 is the csv file where the results should be saved
# e.g. tuning/tuning_results/tuning_run_01.csv

# baseline
python -m tuning.tuning $1 --model_type baseline

# kmeans
for num_clusters in 15 30; do
  for find_num_clusters in "acc" "sil"; do
    python -m tuning.tuning $1 --model_type kmeans --num_clusters $num_clusters --find_num_clusters $find_num_clusters
  done
done

# gmm
for num_components in 15 30; do
  for find_num_clusters in "acc" "bic"; do
    for cov_type in "tied" "diag"; do
      python -m tuning.tuning $1 --model_type gmm --num_components $num_components --find_num_clusters $find_num_clusters --cov_type $cov_type
    done
  done
done

# bmm
for num_components in 15 30; do
  for cov_type in "tied" "diag"; do
    python -m tuning.tuning $1 --model_type bmm --num_components $num_components --cov_type $cov_type
  done
done

# random forest
for n_estimators in 25, 100, 500, 1000; do
  for criterion in "entropy" "gini"; do
    for max_features in "sqrt" "log2"; do
      for max_samples in 0.4 0.6 0.8; do
        for resample in 0 1; do
          python -m tuning.tuning $1 --model_type rf --n_estimators $n_estimators --criterion $criterion --max_features $max_features --max_samples $max_samples --resample $resample
        done
      done
    done
  done
done

# support vector machines
for decision_function_shape in "ovr" "ovo"; do
  for gamma in "auto" "scale"; do
    for kernel in "rbf" "sigmoid"; do
      python -m tuning.tuning $1 --model_type svm --decision_function_shape $decision_function_shape --gamma $gamma --kernel $kernel
    done
  done
done

# k nearest neighbors
for n_neighbors in 10 20 50 100 1000; do
  python -m tuning.tuning $1 --model_type knn --n_neighbors $n_neighbors
done

# easy ensemble
for n_estimators in 10 100 500 1000; do
  for sampling_strategy in "all" "not minority"; do
    python -m tuning.tuning $1 --model_type easy_ensemble --n_estimators $n_estimators --sampling_strategy $sampling_strategy
  done
done

# lstm
for batch_size in 32 8; do
  for rnn_size in 50 100 150; do
    for learning_rate in 0.01 0.001; do
      for dropout in 0 0.2 0.5; do
        for dense_units in 0 100; do
          python -m tuning.tuning $1 --model_type lstm --batch_size $batch_size --epochs 100 --rnn_size $rnn_size --learning_rate $learning_rate --dropout $dropout --dense_units $dense_units
        done
      done
    done
  done
done

# blstm
for batch_size in 32 8; do
  for rnn_size in 50 100 150; do
    for learning_rate in 0.01 0.001; do
      for dropout in 0 0.2 0.5; do
        for dense_units in 0 100; do
          python -m tuning.tuning $1 --model_type blstm --batch_size $batch_size --epochs 100 --rnn_size $rnn_size --learning_rate $learning_rate --dropout $dropout --dense_units $dense_units
        done
      done
    done
  done
done

# encoder decoder
for batch_size in 32 8; do
  for regularizer in 0 1; do
    for learning_rate in 0.001 0.0001; do
      for dropout in 0 0.5; do
        for dense_units in 0 100; do
          for bidirectional in 0 1; do
            for attention in 0 1; do
              python -m tuning.tuning $1 --model_type lstm --batch_size $batch_size --epochs 100 --rnn_size 150 --learning_rate $learning_rate --dropout $dropout --dense_units $dense_units
            done
          done
        done
      done
    done
  done
done

# label spreading
for kernel in "knn" "rbf"; do
  for alpha in 0 0.2 0.4; do
    python -m tuning.tuning $1 --model_type label_spreading --kernel $kernel --alpha $alpha
  done
done
