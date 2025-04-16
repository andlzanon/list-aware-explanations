import cornac
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from dataset_experiment import metrics
from dataset_experiment.dataset_experiment import DatasetExperiment
from explanations.explanation import ExplanationAlgorithm


class HierarchicalClustering(ExplanationAlgorithm):
    def __init__(self, dataset: DatasetExperiment, model: cornac.models.Recommender, method: str, criterion: str,
                 metric: str, n_clusters: int, top_n: int, hitems_per_attr=2, vec_method='binary', random_state=42):
        """
        Hierarchical Clustering explanation algorithm
        :param dataset: dataset used in the recommendation model
        :param model: cornac model used to generate recommendations
        :param method: methods are used to compute the distance between two clusters. It will be used in the scipy
            method linkage.
        :param criterion: The criterion to use in forming flat clusters.
        :param metric: The metric used to compute distance between clusters.
        :param n_clusters: Number of clusters
        :param top_n: number of top attributes to generate the explanation
        :param hitems_per_attr: number of historic items showed per attribute on explanation.
            In an example such as: I recommend you Titanic since you ofter like drama items as X, Y, Z.
            hitems_per_attr is 3, because we are using X, Y and Z profile items to support the attribute
        :param vec_method: Method to transform features into vectors. Can be 'binary' to transform into an array
        with zeros and ones, where the 0 represent that the item does not have the feature and 1 that it does, or
        'relevance' to use the Musto graph relevance score
        :param random_state: random state number for reproducible results
        """
        super().__init__(dataset, model)
        self.method = method
        self.criterion = criterion
        self.metric = metric
        self.n_clusters = n_clusters
        self.top_n = top_n
        self.hitems_per_attr = hitems_per_attr
        self.vec_method = vec_method
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.model_name = (f"Hierarchical&method={str(self.method)}&criterion={str(self.criterion)}"
                            f"&metric={str(self.metric)}&n_clusters={str(self.n_clusters)}&top_n={str(self.top_n)}"
                            f"&hitems_per_attr={str(self.hitems_per_attr)}&vec_method={str(self.vec_method)}"
                            f"&random_state={str(self.random_state)}")

    def user_explanation(self, user: str, top_k: int, remove_seen=True, verbose=True, show_dendrogram=False, **kwargs) \
            -> dict:
        """
        Generate explanation based on hierarchical clustering of items based on the simple presence of those
        We will be able to generate cuts on the dendrogram and generate explanations for the clusters generated
        :param user: user id
        :param top_k: top k items to explain
        :param remove_seen: True if model should exclude seen items, False otherwise
        :param verbose: True to print explanations
        :param show_dendrogram: True to print the generated dendrogram
        :param kwargs: additional arguments
        :return:
        """
        user_explanations = {}
        obj_column = self.dataset.prop_set.columns[-1]
        interacted_items = []
        attributes = []
        ranked_clusters = []

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

        # get all profile and recommended attributes from kg
        pro_all_attr = np.array(list(pro_all_attr))
        rec_all_attr = self.dataset.prop_set.loc[list(map(int, ranked_items))][obj_column]

        # get intersection between interacted and recommended attributes
        inter = set(rec_all_attr).intersection(set(pro_all_attr))
        inter = np.array(list(inter))

        # create clustering dataset based on intersection
        clustering_df = pd.DataFrame(columns=inter)

        for rec_item in ranked_items:
            # initialize vector array of recommended item with all 0 and get recommended attributes
            rec_attr = self.dataset.prop_set.loc[int(rec_item)][obj_column]
            vectorize = np.zeros(inter.shape)

            # create the vector array of a recommended item based on binary presence of attributes
            if self.vec_method == 'binary':
                vectorize = np.isin(inter, rec_attr).astype(int)
            # create the vector array of a recommended item based on TF-IDF relevance score
            elif self.vec_method == 'relevance':
                items_historic = [next((int(k) for k, v in self.dataset.train.iid_map.items() if v == u_item), None)
                                  for u_item in
                                  self.dataset.train.chrono_user_data[self.dataset.train.uid_map[user]][0]]
                prop_dict = self.__user_semantic_profile(items_historic)

                l_rec_attr = list(rec_attr)
                for j in range(0, len(inter)):
                    attr = inter[j]
                    attr_value = 0
                    if attr in l_rec_attr:
                        attr_value = prop_dict[attr]
                    vectorize[j] = attr_value

            clustering_df.loc[len(clustering_df)] = vectorize

        clustering_data = clustering_df.to_numpy()

        # run hierarchical clustering
        linkage_matrix = linkage(clustering_data, method=self.method, metric=self.metric)
        clusters = fcluster(linkage_matrix, t=self.n_clusters, criterion=self.criterion)
        if show_dendrogram: print(clusters)
        for i in range(0, self.n_clusters):
            # get items on cluster, then the attributes of the items on the cluster
            i_cluster = [j for j in range(0, len(clusters)) if clusters[j] == i+1]
            cluster_attr = clustering_df.iloc[i_cluster]
            # generate explanations based on clusters
            if self.vec_method == 'binary':
                # sum the rows to check what attributes are common across all items
                cluster_sum = cluster_attr.sum(axis=0)
                # get arbitrary the top 2 attributes common across all items in the cluster
                expl_attr_names = cluster_sum[cluster_sum == len(i_cluster)].sort_index()
                expl_attr_names = expl_attr_names.sample(frac=1, random_state=self.random_state).index[:self.top_n]
            elif self.vec_method == 'relevance':
                # Keep only columns where all values are different from 0 and take mean
                cluster_nonzero = cluster_attr.loc[:, cluster_attr.ne(0).all(axis=0)]
                expl_attr_names = cluster_nonzero.mean().sort_values(ascending=False).index[:self.top_n]
            else:
                raise ValueError("Parameter vec_method is misspelled or does not exist.")

            # get recommended item names
            rec_item_ids = np.array(ranked_items)[i_cluster].astype(int)
            rec_item_names = self.dataset.prop_set.loc[rec_item_ids]['title'].unique()

            # get profile item names that have the explanation attributes
            pro_df = self.dataset.prop_set.loc[list(pro_items.astype(int))]
            pro_item_ids = pro_df.groupby(pro_df.index)["obj"].apply(set)
            pro_item_ids = pro_item_ids.apply(lambda attrs: set(attrs).issuperset(set(expl_attr_names)))
            pro_item_ids = pro_item_ids[pro_item_ids == True].index.astype(int)
            pro_item_ids = np.random.choice(pro_item_ids,
                                              size=pro_item_ids.shape[0], replace=False)[:self.hitems_per_attr]
            pro_item_names = self.dataset.prop_set.loc[pro_item_ids]['title'].unique().tolist()

            interacted_items.append(pro_item_ids)
            attributes.append(expl_attr_names)

            # now we have all elements, lets create the sentence:
            if len(pro_item_names) > 0:
                expl = f'''If you are in the mood for {", ".join(expl_attr_names)} items such as 
                    {", ".join(list(pro_item_names))}, I recommend {", ".join(rec_item_names)}\n'''
            else:
                expl = f'''If you are in the mood for {", ".join(expl_attr_names)} items,
                 I recommend {", ".join(rec_item_names)}\n'''

            if verbose: print(expl)
            for rec in rec_item_ids:
                user_explanations[rec] = expl

        if show_dendrogram:
            plt.figure(figsize=(16, 8))
            dendrogram(linkage_matrix)
            plt.title("Dendrogram")
            plt.xlabel("Samples")
            plt.ylabel("Distance")
            plt.xticks(fontsize=6, rotation=90)
            plt.legend()
            plt.show()

        hier_metrics = metrics.hierarchical_clustering_metrics(linkage_matrix, verbose=False)
        clu_metrics = metrics.clustering_metrics(clustering_data, clusters, verbose=False)
        item_cluster_metrics = metrics.items_per_cluster(clusters.tolist())

        unique_items = list(set([item for sublist in interacted_items for item in sublist]))
        unique_attributes = list(set([item for sublist in attributes for item in sublist]))
        lir = metrics.lir_metric(beta=0.3, user=user, items=unique_items,
                                 train_set=self.dataset.load_fold_asdf()[0],
                                 col_user=self.dataset.user_column)
        sep = metrics.sep_metric(beta=0.3, props=attributes, prop_set=self.dataset.prop_set, memo_sep=self.memo_sep)
        etd = metrics.etd_metric(unique_attributes, top_k, len(self.dataset.prop_set['obj'].unique()))

        attr_metrics = {
            "SEP": sep,
            "LIR": lir,
            "ETD": etd
        }

        # generate re-ranking based on clustering
        retrieved_cluster = []
        for i in range(0, len(ranked_items)):
            item_cluster = clusters[i]
            if item_cluster not in retrieved_cluster:
                retrieved_cluster.append(item_cluster)
                cluster_indexes = [i for i, n in enumerate(clusters) if n == item_cluster]
                ranked_clusters.append([ranked_items[cluster_indexes[i]] for i in range(0, len(cluster_indexes))])

        rerank = pd.DataFrame(columns=[self.dataset.user_column, self.dataset.item_column, "x_rank", "y_rank"])
        matrix = pd.DataFrame(ranked_clusters).to_numpy()
        for x in range(0, matrix.shape[0]):
            for y in range(0, matrix.shape[1]):
                item = matrix[x][y]
                if item is not None:
                    rerank.loc[len(rerank)] = [user, item, x+1, y+1]

        expl_metrics = {
            "hierarchical_metrics": hier_metrics,
            "items_cluster_metrics": item_cluster_metrics,
            "cluster_metrics": clu_metrics,
            "attribute_metrics": attr_metrics
        }

        ret_obj = {
            "grid_items": rerank,
            "explanations": user_explanations,
            "clusters": clusters,
            "metrics": expl_metrics
        }
        return ret_obj

    def all_users_explanations(self, top_k: int, output_file: str, remove_seen=True, verbose=True) -> None:
        # TODO: implement function
        pass

    def __user_semantic_profile(self, historic: list) -> dict:
        """
        Generate the user semantic profile, where all the values of properties (e.g.: George Lucas, action films, etc)
        are ordered by a score that is calculated as:
            score = (npi/i) * log(N/dft)
        where npi are the number of edges to a value, i the number of interacted items,
        N the total number of items and dft the number of items with the value
        :param historic: list of the items interacted by a user
        :return: dictionary with properties' values as keys and scores as values
        """

        # create npi, i and n columns
        interacted_props = self.dataset.prop_set.loc[self.dataset.prop_set.index.isin(historic)].copy()
        interacted_props['npi'] = interacted_props.groupby(self.dataset.prop_set.columns[-1])[self.dataset.prop_set.columns[-1]].transform(
            'count')
        interacted_props['i'] = len(historic)
        interacted_props['n'] = len(self.dataset.prop_set.index.unique())

        # get items per property on full dbpedia/wikidata by dropping the duplicates with same item id and prop value
        # therefore, a value that repeats in the same item is ignored
        items_per_obj = self.dataset.prop_set.reset_index().drop_duplicates(
            subset=[self.dataset.prop_set.columns[0], self.dataset.prop_set.columns[-1]]).set_index(
            self.dataset.prop_set.columns[-1])
        df_dict = items_per_obj.index.value_counts().to_dict()

        # generate the dft column based on items per property and score column base on all new created columns
        interacted_props['dft'] = interacted_props.apply(lambda x: df_dict[x[self.dataset.prop_set.columns[-1]]], axis=1)

        interacted_props['score'] = (interacted_props['npi'] / interacted_props['i']) * (
            np.log(interacted_props['n'] / interacted_props['dft']))

        # generate the dict
        interacted_props.reset_index(inplace=True)
        interacted_props = interacted_props.set_index(self.dataset.prop_set.columns[-1])
        fav_prop = interacted_props['score'].to_dict()

        return fav_prop
