from data_preprocessing import npz_to_pd, idx_to_int

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# labels for the different grain type markers
LABELS = {"not_labelled": 0, "surface": 1, "ground": 2, "dh": 3, "dhid": 4, "mfdh": 5, "rgwp": 6,
          "df": 7, "if": 8, "ifwp": 9, "sh": 10, "drift_end": 11, "snow-ice": 12}

def main():
    # load dataframe with smp data
    smp = npz_to_pd("smp_all_final.npz", is_dir=False)

    # toy example - choose one labelled datapoint
    sample = smp[smp["smp_idx"] == idx_to_int("S31H0369")]
    print(sample.head())

    # see how the data looks like
    sns.lineplot(sample["distance"], sample["mean_force"]).set_title("Distance and Mean force of S31H0369")
    plt.show()

    sns.scatterplot(sample["distance"], sample["mean_force"], hue=sample["label"]).set_title("Distance and Mean force with labels of S31H0369")
    plt.show()

    sns.scatterplot(sample["var_force"], sample["mean_force"], hue=sample["label"]).set_title("Variance and Mean force of S31H0369")
    plt.show()

    # k-means clustering for one sample
    km = KMeans(n_clusters=5, init="random", n_init=10, random_state=42)

    clusters = km.fit_predict(sample[["mean_force", "var_force"]])
    print(clusters)

    sns.scatterplot(sample["var_force"], sample["mean_force"], hue=sample["label"], style=clusters).set_title("Clustering of S31H0369")
    plt.show()

    # take a look on the big data!

    # more data
    smp_more = smp.sample(n=2000, random_state=42)
    sns.scatterplot(smp_more["var_force"], smp_more["mean_force"]).set_title("Variance and Mean force 2000 random data points")
    plt.show()

    # we have in total 10 labels
    km_more = KMeans(n_clusters=10, init="random", n_init=100, random_state=42)

    clusters = km_more.fit_predict(smp_more[["mean_force", "var_force"]])
    print(clusters)

    sns.scatterplot(smp_more["var_force"], smp_more["mean_force"], hue=clusters).set_title("Variance and Mean force of 1000 samples")
    plt.show()


    # only labelled data
    smp_labelled = smp[smp["label"] != 0]

    sns.scatterplot(smp_labelled["var_force"], smp_labelled["mean_force"], hue=smp_labelled["label"]).set_title("Variance and Mean force for all labelled data")
    plt.show()

    # k-means clustering for all which are labelled

    # we have in total 10 labels
    km_lab = KMeans(n_clusters=10, init="random", n_init=100, random_state=42)

    clusters = km_lab.fit_predict(smp_labelled[["mean_force", "var_force"]])
    print(clusters)

    sns.scatterplot(smp_labelled["var_force"], smp_labelled["mean_force"], hue=smp_labelled["label"], style=clusters).set_title("Clustering for labelled data")
    plt.show()




if __name__ == "__main__":
    main()
