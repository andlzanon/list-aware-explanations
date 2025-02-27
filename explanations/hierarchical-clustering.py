import cornac
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from dataset_experiment.dataset_experiment import DatasetExperiment
from explanations.explanation import ExplanationAlgorithm


class HierarchicalClustering(ExplanationAlgorithm):
    def __init__(self, dataset: DatasetExperiment, model: cornac.models.Recommender, method: str, criterion: str,
                 n_clusters: int):
        """
        Hierarchical Clustering explanation algorithm
        :param dataset: dataset used in the recommendation model
        :param model: cornac model used to generate recommendations
        :param method: methods are used to compute the distance between two clusters. It will be used in the scipy
            method linkage.
        :param criterion: The criterion to use in forming flat clusters.
        :param n_clusters: Number of clusters
        """
        super().__init__(dataset, model)
        self.method = method
        self.criterion = criterion
        self.n_clusters = n_clusters

    def user_explanation(self, user: str, top_k: int, remove_seen=True, verbose=True, **kwargs) -> dict:
        user_explanations = {}

        # generate user recommendations
        ranked_items = list(self.model.recommend(user_id=user, k=top_k,
                                                 train_set=self.dataset.train,
                                                 remove_seen=remove_seen))

        #  get train dataset and set user as index
        train_df = self.dataset.load_fold_asdf()[0]
        train_df = train_df.set_index(self.dataset.user_column)

        # create a set of all profile attributes
        pro_all_attr = set()
        pro_items = train_df.loc[user][self.dataset.item_column]

        # for evey user profile item get its properties
        for pro_item in pro_items:
            i_attr = self.dataset.prop_set[pro_item][self.dataset.prop_set.columns[-1]]
            pro_all_attr = pro_all_attr.union(set(list(i_attr)))

        # creating test dataset with presence of attributes
        pro_all_attr = np.array(list(pro_all_attr))
        clustering_df = pd.DataFrame(columns=pro_all_attr)

        # binarize the presence of attributes on all recommended items based on profile items attributes
        for rec_item in ranked_items:
            rec_attr = self.dataset.prop_set[rec_item][self.dataset.prop_set.columns[-1]]
            if len(rec_attr) > len(pro_all_attr):
                vectorize = np.isin(rec_attr, pro_all_attr).astype(int)
            else:
                vectorize = np.isin(pro_all_attr, rec_attr).astype(int)

            clustering_df.loc[len(clustering_df)] = vectorize

        clustering_data = clustering_df.to_numpy()

        # run hierarchical clustering
        linkage_matrix = linkage(clustering_data, method=self.method)
        clusters = fcluster(linkage_matrix, t=self.n_clusters, criterion=self.criterion)

        if verbose:
            plt.figure(figsize=(16, 8))
            dendrogram(linkage_matrix)
            plt.title("Dendrogram")
            plt.xlabel("Samples")
            plt.ylabel("Distance")
            plt.xticks(fontsize=6, rotation=90)
            plt.legend()
            plt.show()

        # generate explanations based on clusters

        return user_explanations

    def all_users_explanations(self, top_n: int, output_file: str, remove_seen=True, verbose=True) -> None:
        pass
