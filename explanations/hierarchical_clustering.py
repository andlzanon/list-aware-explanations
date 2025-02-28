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
        """
        Generate explanation based on hierarchical clustering of items based on the simple presence of those
        We will be able to generate cuts on the dendrogram and generate explanations for the clusters generated
        :param user: user id
        :param top_k: top k items to explain
        :param remove_seen: True if model should exclude seen items, False otherwise
        :param verbose: True to print explanations
        :param kwargs: additional arguments
        :return:
        """
        user_explanations = {}
        obj_column = self.dataset.prop_set.columns[-1]

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
            i_attr = self.dataset.prop_set.loc[int(pro_item)][self.dataset.prop_set.columns[-1]]
            pro_all_attr = pro_all_attr.union(set(list(i_attr)))

        # creating test dataset with presence of attributes
        pro_all_attr = np.array(list(pro_all_attr))
        clustering_df = pd.DataFrame(columns=pro_all_attr)

        # binarize the presence of attributes on all recommended items based on profile items attributes
        for rec_item in ranked_items:
            rec_attr = self.dataset.prop_set.loc[int(rec_item)][obj_column]
            if len(rec_attr) > len(pro_all_attr):
                vectorize = np.isin(rec_attr, pro_all_attr).astype(int)
            else:
                vectorize = np.isin(pro_all_attr, rec_attr).astype(int)

            clustering_df.loc[len(clustering_df)] = vectorize

        clustering_data = clustering_df.to_numpy()

        # run hierarchical clustering
        linkage_matrix = linkage(clustering_data, method=self.method)
        clusters = fcluster(linkage_matrix, t=self.n_clusters, criterion=self.criterion)
        print(clusters)
        for i in range(0 , self.n_clusters):
            # get items on cluster, then the attributes of the items on the cluster
            i_cluster = [j for j in range(0, len(clusters)) if clusters[j] == i+1]
            cluster_attr = clustering_df.iloc[i_cluster]
            # sum the rows to check what attributes are common across all items
            cluster_sum = cluster_attr.sum(axis=0)
            # get arbitrary the top 2 attributes common across all items in the cluster
            # TODO: get by popularity or other criteria
            expl_attr_names = cluster_sum[cluster_sum == len(i_cluster)].index[:2]

            # get recommended item names
            rec_item_ids = np.array(ranked_items)[i_cluster].astype(int)
            rec_item_names = self.dataset.prop_set.loc[rec_item_ids]['title'].unique()

            # get profile item names that have the explanation attributes
            pro_df = self.dataset.prop_set.loc[list(pro_items.astype(int))]
            pro_item_ids = pro_df.groupby(level=0)[obj_column].apply(lambda x: set(expl_attr_names).issubset(set(x)))
            pro_item_ids = pro_item_ids[pro_item_ids == True].index.astype(int)
            pro_item_names = self.dataset.prop_set.loc[pro_item_ids]['title'].unique()[:2]

            # now we have all elements, lets create the sentence:
            if pro_item_names.shape[0] > 0:
                expl = f'''If you are in the mood for {", ".join(expl_attr_names)} items such as 
                    {", ".join(list(pro_item_names))}, I recommend {", ".join(rec_item_names)}\n'''
            else:
                expl = f'''If you are in the mood for {", ".join(expl_attr_names)} items,
                 I recommend {", ".join(rec_item_names)}\n'''

            print(expl)
            for rec in rec_item_ids:
                user_explanations[rec] = expl

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
        # TODO: implement function
        pass
