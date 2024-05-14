if __name__ == "__main__":
    import configparser
    import json
    import datetime
    import os

    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from utility import *

    CONFIG = "../../config/config_mesh_static.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    # data
    read_data = config["DATA"]
    data_dir = read_data["data_dir"]
    # save setting
    read_save = config["SAVE"]
    output_dir = read_save["output_dir"]
    fig_dir = read_save["figure_dir"]
    image_file = os.path.join(fig_dir, "train.png")

    target_case = "20240514105617"
    output_dir = os.path.join(output_dir, target_case)

    latent_dir = os.path.join(output_dir, "latent")

    file_names = []
    labels = None  # (output_channels * num_file)
    features = None  # (output_channels * num_file, emb_dim)
    feature_cat = None  # (num_file, output_channels * emb_dim)
    for cur_dir, dirs, files in os.walk(latent_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".txt":
                latent_feature = load_2d_ndarray(os.path.join(cur_dir, file))  # (output_channel, emb_dim)
                label = np.arange(latent_feature.shape[0])

                if labels is None:
                    labels = label
                    features = latent_feature
                    feature_cat = latent_feature.reshape(1, -1)
                else:
                    labels = np.concatenate((labels, label), axis=0)
                    features = np.concatenate((features, latent_feature), axis=0)
                    feature_cat = np.concatenate((latent_feature.reshape(1, -1), feature_cat), axis=0)
                file_names.append(file)
    output_channel = max(labels) + 1
    print(output_channel)

    dbs = []  # (output_channel)
    pred_labels = []  # (output_channel, num_files)
    for i in range(output_channel):
        tmp_feature = features[labels == i]
        tmp_feature = StandardScaler().fit_transform(tmp_feature)
        db = DBSCAN(eps=1.0, min_samples=10).fit(tmp_feature)
        pred_label = db.labels_

        dbs.append(db)
        pred_labels.append(pred_label)

        n_clusters_ = len(set(pred_labels[-1])) - (1 if -1 in pred_labels[-1] else 0)
        n_noise_ = list(pred_labels[-1]).count(-1)
        print(f"Output_channel {i}")
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        print("")

    feature_cat = StandardScaler().fit_transform(feature_cat)
    feature_pca = PCA(2).fit_transform(feature_cat)

    pred_labels = np.array(pred_labels)

    val = np.concatenate((np.expand_dims(np.array(file_names), axis=1), pred_labels.T), axis=1)
    df_label = pd.DataFrame(val, columns=["file", *[f"channel {i}" for i in range(output_channel)]])

    clusters = df_label[[f"channel {i}" for i in range(output_channel)]].drop_duplicates()
    clusters = {tuple(clusters.loc[idx, :]): idx for idx in clusters.index}
    idx2cluster = {v:k for k, v in clusters.items()}
    df_label["cluster"] = [clusters[tuple(df_label.loc[idx, [f"channel {i}" for i in range(output_channel)]])] for idx in df_label.index]

    df_label.to_csv(os.path.join(output_dir, "cluster.csv"), index=False)

    for idx, clu in idx2cluster.items():
        #if "-1" in clu:
        #    continue
        tmp_idx = df_label["cluster"] == idx
        plt.scatter(feature_pca[tmp_idx, 0], feature_pca[tmp_idx, 1], label=str(clu))
    plt.legend()
    plt.xlabel("PCA 0")
    plt.ylabel("PCA 1")
    plt.savefig(os.path.join(output_dir, "cluster.png"))
    plt.close()
    
