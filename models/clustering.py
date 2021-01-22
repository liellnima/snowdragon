# import from other snowdragon modules
from data_handling.data_loader import load_data
from data_handling.data_preprocessing import idx_to_int
from data_handling.data_parameters import LABELS

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# labels for the different grain type markers (just for visibility listed here)
LABELS = {"not_labelled": 0, "surface": 1, "ground": 2, "dh": 3, "dhid": 4, "mfdh": 5, "rgwp": 6,
          "df": 7, "if": 8, "ifwp": 9, "sh": 10, "drift_end": 11, "snow-ice": 12}

ANTI_LABELS = {0: "not_labelled",  1: "surface", 2: "ground", 3: "dh", 4: "dhid", 5: "mfdh", 6: "rgwp",
          7: "df", 8: "if", 9: "ifwp", 10:"sh", 11: "drift_end", 12: "snow-ice"}

COLORS = {0: "lightsteelblue", 1: "chocolate", 2: "darkslategrey", 3: "lightseagreen", 4: "mediumaquamarine", 5: "midnightblue",
          6: "tomato", 7: "mediumvioletred", 8: "firebrick", 9: "lightgreen", 10: "orange", 11: "black", 12: "paleturquoise"}

# TODO put this in different functions (for re-usage!)
# TODO maybe even put this in a completely different file for visualization purposes
def visualize_original_data(smp):
    """ Visualizing some things of the original data
    """
    # # HOW BALANCED IS THE LABELLED DATASET?
    #
    # # take only labelled data and exclude surface and ground
    # labelled_smp = smp[(smp["label"] != 0) & (smp["label"] != 1) & (smp["label"] != 2)]
    # # I can currently not find snow-ice because I am cutting of the last datapoints, if they are less than 1mm
    # # TODO: do not cut off last part during summarizing rows -> to get snow-ice I have to fix this!
    # print("Can I find the snow-ice label?", smp[smp["label"] == 12])
    #
    # ax = sns.countplot(x="label", data=labelled_smp, order=labelled_smp["label"].value_counts().index)
    # plt.title("Distribution of Labels in the Labelled SMP Dataset")
    # plt.xlabel("Labels")
    # ax2=ax.twinx()
    # ax2.set_ylabel("Frequency [%]")
    # for p in ax.patches:
    #     x=p.get_bbox().get_points()[:,0]
    #     y=p.get_bbox().get_points()[1,1]
    #     ax.annotate("{:.1f}%".format(100.*y/len(labelled_smp)), (x.mean(), y), ha="center", va="bottom") # set the alignment of the text
    # x_labels = [ANTI_LABELS[label_number] for label_number in labelled_smp["label"].value_counts().index]
    # ax.set_xticklabels(x_labels, rotation=90)
    # ax2.set_ylim(0,100)
    # ax.set_ylim(0,len(labelled_smp))
    # plt.show()
    #
    # # SHOW ONE SMP PROFILE WITHOUT LABELS
    # smp_profile_name = "S31H0368" #"S31H0607"
    # smp_profile = smp[smp["smp_idx"] == idx_to_int(smp_profile_name)]
    # ax = sns.lineplot(smp_profile["distance"], smp_profile["mean_force"])
    # plt.title("{} SMP Profile Distance (1mm layers) and Force".format(smp_profile_name))
    # ax.set_xlabel("Snow Depth [mm]")
    # ax.set_ylabel("Mean Force [N]")
    # plt.show()
    #
    # # SHOW THE SAME PROFILE WITH LABELS
    #
    # ax = sns.lineplot(smp_profile["distance"], smp_profile["mean_force"])
    # plt.title("{} SMP Profile Distance (1mm layers) and Force".format(smp_profile_name))
    #
    # used_labels=[]
    # last_label_num = 1
    # last_distance = -1
    # # going through labels and distance
    # for label_num, distance in zip(smp_profile["label"], smp_profile["distance"]):
    #     if (label_num != last_label_num):
    #         # assign new background for each label
    #         background = ax.axvspan(last_distance, distance-1, color=COLORS[last_label_num], alpha=0.5)
    #         # set labels for legend
    #         if ANTI_LABELS[last_label_num] not in used_labels:
    #             background.set_label(ANTI_LABELS[last_label_num])
    #             used_labels.append(ANTI_LABELS[last_label_num])
    #
    #         last_label_num = label_num
    #         last_distance = distance-1
    #
    #     if distance == smp_profile.iloc[len(smp_profile)-1]["distance"]:
    #         ax.axvspan(last_distance, distance, color=COLORS[label_num], alpha=0.5).set_label(ANTI_LABELS[last_label_num])
    #
    # ax.legend()
    # ax.set_xlabel("Snow Depth [mm]")
    # ax.set_ylabel("Mean Force [N]")
    # plt.show()

    # PLOT ALL FEATURES AS LINES IN ONE PROFILE

    # SHOW THE DATADISTRIBUTION OF ALL FEATURES

    # SHOW HEATMAP OF ALL FEATURES (with what are the labels correlated the most?)
    #https://stackoverflow.com/questions/37790429/seaborn-heatmap-using-pandas-dataframe
    # sns.heatmap([smp["label"], smp["distance"]])
    # plt.show()

def main():
    # load dataframe with smp data
    smp = load_data("smp_lambda_delta_gradient.npz")
    visualize_original_data(smp)

    # # toy example - choose one labelled datapoint
    # sample = smp[smp["smp_idx"] == idx_to_int("S31H0369")]
    # print(sample.head())
    #
    # # see how the data looks like
    # sns.lineplot(sample["distance"], sample["mean_force"]).set_title("Distance and Mean force of S31H0369")
    # plt.show()
    #
    # sns.scatterplot(sample["distance"], sample["mean_force"], hue=sample["label"]).set_title("Distance and Mean force with labels of S31H0369")
    # plt.show()
    #
    # sns.scatterplot(sample["var_force"], sample["mean_force"], hue=sample["label"]).set_title("Variance and Mean force of S31H0369")
    # plt.show()
    #
    # # k-means clustering for one sample
    # km = KMeans(n_clusters=5, init="random", n_init=10, random_state=42)
    #
    # clusters = km.fit_predict(sample[["mean_force", "var_force"]])
    # print(clusters)
    #
    # sns.scatterplot(sample["var_force"], sample["mean_force"], hue=sample["label"], style=clusters).set_title("Clustering of S31H0369")
    # plt.show()
    #
    # # take a look on the big data!
    #
    # # more data
    # smp_more = smp.sample(n=2000, random_state=42)
    # sns.scatterplot(smp_more["var_force"], smp_more["mean_force"]).set_title("Variance and Mean force 2000 random data points")
    # plt.show()
    #
    # # we have in total 10 labels
    # km_more = KMeans(n_clusters=10, init="random", n_init=100, random_state=42)
    #
    # clusters = km_more.fit_predict(smp_more[["mean_force", "var_force"]])
    # print(clusters)
    #
    # sns.scatterplot(smp_more["var_force"], smp_more["mean_force"], hue=clusters).set_title("Variance and Mean force of 1000 samples")
    # plt.show()
    #
    #
    # # only labelled data
    # smp_labelled = smp[smp["label"] != 0]
    #
    # sns.scatterplot(smp_labelled["var_force"], smp_labelled["mean_force"], hue=smp_labelled["label"]).set_title("Variance and Mean force for all labelled data")
    # plt.show()
    #
    # # k-means clustering for all which are labelled
    #
    # # we have in total 10 labels
    # km_lab = KMeans(n_clusters=10, init="random", n_init=100, random_state=42)
    #
    # clusters = km_lab.fit_predict(smp_labelled[["mean_force", "var_force"]])
    # print(clusters)
    #
    # sns.scatterplot(smp_labelled["var_force"], smp_labelled["mean_force"], hue=smp_labelled["label"], style=clusters).set_title("Clustering for labelled data")
    # plt.show()




if __name__ == "__main__":
    main()
