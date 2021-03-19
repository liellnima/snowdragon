# baseline
python -m tuning.tuning tuning/tuning_results/tuning_run_01.csv --model_type baseline

# kmeans
for num_clusters in 15 30; do
  for find_num_clusters in "acc" "sil"; do
    python -m tuning.tuning tuning/tuning_results/tuning_run_01.csv --model_type kmeans --num_clusters $num_clusters --find_num_clusters $find_num_clusters
  done
done
