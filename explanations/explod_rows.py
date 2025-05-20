import os

import cornac
import numpy as np
import pandas as pd

from dataset_experiment import metrics
from dataset_experiment.dataset_experiment import DatasetExperiment
from dataset_experiment.metrics import fill_ideal_grid_by_manhattan
from explanations.explanation import ExplanationAlgorithm


class ExpLODRows(ExplanationAlgorithm):
    def __init__(self, dataset: DatasetExperiment, model: cornac.models.Recommender, expr_file: str, top_k: int,
                 top_n=1, hitems_per_attr=2, n_users=0, alpha=0.5, beta=0.5, random_state=42, n_clusters=None):
        """
        ExpLOD algorithm as in https://dl.acm.org/doi/abs/10.1145/2959100.2959173
        :param dataset: dataset used in the recommendation model
        :param model: cornac model used to generate recommendations
        :param expr_file: name of the experiment file configuration
        :param top_k: top k items to explain
        :param top_n: number of top attributes to generate the explanation
        :param hitems_per_attr: number of historic items showed per attribute on explanation.
            In an example such as: I recommend you Titanic since you ofter like drama items as X, Y, Z.
            hitems_per_attr is 3, because we are using X, Y and Z profile items to support the attribute
        :param n_users: number of users to generate explanations to. If 0 runs to all users
        :param alpha: weight to the number of links of attributes on interacted items
        :param beta: weight on number of links of attributes to recommended items
        :param n_clusters: Number of clusters
        """
        super().__init__(dataset, model, expr_file, top_k, n_users)
        self.top_n = top_n
        self.hitems_per_attr = hitems_per_attr
        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_clusters
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.model_name = (f"ExpLODRows&top_n={str(self.top_n)}&hitems_per_attr={str(self.hitems_per_attr)}"
                           f"&top_k={str(self.top_k)}&alpha={str(self.alpha)}&beta={str(self.beta)}"
                           f"&rs={str(self.random_state)}"
                           f"&n_clusters={str(self.n_clusters)}&u={str(abs(self.n_users))}")
        self.expl_file_path = r"\\?\\" + os.path.abspath(self.expl_file_path + self.model_name + ".txt")
        open(self.expl_file_path, 'w+').close()

    def __user_semantic_profile(self, historic: list, recommended: list) -> dict:
        """
        Generate the user semantic profile, where all the values of properties (e.g.: George Lucas, action films, etc.)
        are ordered by a score that is calculated as:
            score = (npi/i + npr/r) * log(N/dft)
        where npi are the number of edges to a value, i the number of interacted items,
        N the total number of items and dft the number of items with the value
        :param historic: list of the items interacted by a user
        :param recommended: list of the items recommended to the user
        :return: dictionary with properties' values as keys and scores as values
        """
        # create npi, i and n columns
        interacted_props = self.dataset.prop_set.loc[self.dataset.prop_set.index.isin(historic)].copy()
        interacted_props['npi'] = interacted_props.groupby(self.dataset.prop_set.columns[-1])[
            self.dataset.prop_set.columns[-1]].transform('count')
        interacted_props['i'] = len(historic)

        recommended_props = self.dataset.prop_set.loc[self.dataset.prop_set.index.isin(recommended)].copy()
        recommended_props['npr'] = recommended_props.groupby(self.dataset.prop_set.columns[-1])[
            self.dataset.prop_set.columns[-1]].transform('count')
        recommended_props['r'] = len(recommended)

        merged = interacted_props.merge(recommended_props, on='obj', how='inner')
        merged['n'] = len(self.dataset.prop_set.index.unique())

        # get items per property on full dbpedia/wikidata by dropping the duplicates with same item id and prop value
        # therefore, a value that repeats in the same item is ignored
        items_per_obj = self.dataset.prop_set.reset_index().drop_duplicates(
            subset=[self.dataset.prop_set.columns[0], self.dataset.prop_set.columns[-1]]).set_index(
            self.dataset.prop_set.columns[-1])
        df_dict = items_per_obj.index.value_counts().to_dict()

        # generate the dft column based on items per property and score column base on all new created columns
        merged['dft'] = merged.apply(lambda x: df_dict[x[self.dataset.prop_set.columns[-1]]], axis=1)
        merged['idfc'] = np.log(merged['n'] / merged['dft'])

        merged['ncip'] = merged['npi'] / merged['i']
        merged['ncir'] = merged['npr'] / merged['r']

        merged['score'] = (((self.alpha * merged['ncip']) + (self.beta * merged['ncir']))
                                     * merged['idfc'])

        # generate the dict
        merged.reset_index(inplace=True)
        merged = merged.set_index(self.dataset.prop_set.columns[-1])
        fav_prop = merged['score'].to_dict()

        return fav_prop

    def user_explanation(self, user: str, remove_seen=True, verbose=True, **kwargs) -> dict:
        """
        Generate user explanation with ExpLOD algorithm link: https://dl.acm.org/doi/abs/10.1145/2959100.2959173
        :param user: user id
        :param remove_seen: True if model should exclude seen items, False otherwise
        :param verbose: True to print sentences
        :return: explanations as dict where key is recommended item and value is explanation
        """
        user_explanations = {}
        interacted_items = []
        attributes = []
        clusters = []
        rerank_df = pd.DataFrame()
        misses = 0

        item_col = self.dataset.item_column

        # generate user recommendations, this time as tuples
        recommendations = list(self.model.recommend(user_id=user, k=self.top_k,
                                                 train_set=self.dataset.train,
                                                 remove_seen=remove_seen))
        ranked_items = pd.DataFrame(recommendations, columns=[item_col])

        ranked_items_2d = fill_ideal_grid_by_manhattan(ranked_items, rows=self.n_clusters, cols=None)

        max_value = ranked_items_2d['x_irank'].max()
        min_value = ranked_items_2d['y_irank'].min()

        items_historic = [next((int(k) for k, v in self.dataset.train.iid_map.items() if v == u_item), None)
                          for u_item in self.dataset.train.chrono_user_data[self.dataset.train.uid_map[user]][0]]

        with open(self.expl_file_path, 'a+', encoding='utf-8') as f:
            f.write(f'''--- Explanations User Id {user} ---\n''')
        if verbose: print(f'''--- Explanations User Id {user} ---''')

        all_items = list(set(items_historic).union(set(recommendations)))
        all_props = self.dataset.prop_set.loc[self.dataset.prop_set.index.isin(all_items)].copy()['obj']
        all_props = all_props.drop_duplicates().sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        clustering_df = pd.DataFrame(columns=all_props)

        for row in range(min_value, max_value+1):
            rec_row = ranked_items_2d[ranked_items_2d["x_irank"] == row][item_col].astype(int).tolist()

            props = self.__user_semantic_profile(items_historic, rec_row)
            props_sorted = sorted(props.items(), key=lambda item: item[1], reverse=True)
            max_props = [k for k, _ in props_sorted[:self.top_n]]

            for rec in rec_row:
                vectorize = np.zeros(all_props.shape)
                rec_props = self.dataset.prop_set.loc[int(rec)]['obj'].tolist()
                for i in range(0, len(all_props)):
                    prop_name = all_props.iloc[i]
                    attr_value = 0
                    if prop_name in rec_props:
                        attr_value = props[prop_name]
                    vectorize[i] = attr_value
                clustering_df.loc[len(clustering_df)] = vectorize
                clusters.append(row)

            expl_attr_names = []
            for pi in range(0, len(max_props)):
                p = max_props[pi]
                expl_attr_names.append(p)

            # get profile item names that have the explanation attributes
            pro_df = self.dataset.prop_set.loc[items_historic]
            pro_item_ids = pro_df.groupby(pro_df.index)["obj"].apply(set)
            pro_item_ids = pro_item_ids.apply(lambda attrs: set(attrs).issuperset(set(expl_attr_names)))
            pro_item_ids = pro_item_ids[pro_item_ids == True].index.astype(int)
            pro_item_ids = np.random.choice(pro_item_ids,
                                                size=pro_item_ids.shape[0], replace=False)[:self.hitems_per_attr]
            pro_item_names = self.dataset.prop_set.loc[pro_item_ids]['title'].unique().tolist()
            rec_item_names = self.dataset.prop_set.loc[rec_row]['title'].unique()

            interacted_items.append(pro_item_ids)
            attributes.append(expl_attr_names)

            rerank = ranked_items_2d[ranked_items_2d["x_irank"] == row]
            rerank.rename(columns={'x_irank': 'x_rank', 'y_irank': 'y_rank', 'rec_item': item_col}, inplace=True)
            rerank[self.dataset.user_column] = user
            rerank_df = pd.concat([rerank_df, rerank], ignore_index=True)

            # now we have all elements, lets create the sentence:
            if len(pro_item_names) > 0 and len(expl_attr_names) > 0:
                expl = (f"If you are in the mood for {", ".join(expl_attr_names)} items such as "
                        f"{", ".join(list(pro_item_names))}, I recommend {", ".join(rec_item_names)}\n")
            elif len(pro_item_names) == 0 and len(expl_attr_names) > 0:
                expl = (f"If you are in the mood for {", ".join(expl_attr_names)} items, "
                        f"I recommend {", ".join(rec_item_names)}\n")
            elif len(pro_item_names) > 0 and len(expl_attr_names) == 0:
                expl = (f"If you are in the mood for items, items such as "
                        f"{", ".join(list(pro_item_names))}, I recommend {", ".join(rec_item_names)}\n")
                misses = misses + 1
            else:
                raise AttributeError("Profile items array and shared attributes array lengths are 0")

            if verbose: print(expl)
            with open(self.expl_file_path, 'a+', encoding='utf-8') as f:
                f.write(expl)
            for rec in recommendations:
                user_explanations[rec] = expl

        unique_items = list(set([item for sublist in interacted_items for item in sublist]))
        unique_attributes = list(set([item for sublist in attributes for item in sublist]))
        total_attributes = sum([len(sublist) for sublist in attributes])
        total_items = sum([len(sublist) for sublist in interacted_items])

        mid = np.array([len(sublist) for sublist in interacted_items]).mean()
        lir = metrics.lir_metric(beta=0.3, user=user, items=unique_items,
                                 train_set=self.dataset.load_fold_asdf()[0],
                                 col_user=self.dataset.user_column, col_item=self.dataset.item_column)
        sep = metrics.sep_metric(beta=0.3, props=attributes, prop_set=self.dataset.prop_set, memo_sep=self.memo_sep)
        etd = metrics.etd_metric(unique_attributes, self.top_k, total_attributes)
        overlap_attributes = len(unique_attributes) / total_attributes
        try:
            overlap_items = len(unique_items) / total_items
        except ZeroDivisionError:
            # it will be ignored in the metric
            overlap_items = -1

        clustering_data = clustering_df.to_numpy()
        clu_metrics = metrics.clustering_metrics(clustering_data, clusters, verbose=False)
        item_cluster_metrics = metrics.items_per_cluster(ranked_items_2d["x_irank"].astype(int).tolist())

        attr_metrics = {
            "SEP": sep,
            "LIR": lir,
            "ETD": etd,
            "TID": unique_items,
            "TPD": unique_attributes,
            "MID": mid,
            "Overlap-Attributes": overlap_attributes,
            "Overlap-Items": overlap_items,
            "Path-Misses": misses
        }

        expl_metrics = {
            "items_cluster_metrics": item_cluster_metrics,
            "cluster_metrics": clu_metrics,
            "attribute_metrics": attr_metrics
        }

        ret_obj = {
            "grid_items": rerank_df,
            "explanations": user_explanations,
            "clusters": clusters,
            "metrics": expl_metrics
        }

        return ret_obj

    def all_users_explanations(self, remove_seen=True, verbose=True) -> tuple[dict, dict]:
        """
        Method to run explanations to all users and extract explanation metrics
        :param remove_seen: remove seen items on evaluation
        :param verbose: True to display log, False otherwise
        :return: tuple of two dictionaries: one containing the metrics and the other one with all outputs of all users.
        """
        ret_obj = {
            "grid_items": pd.DataFrame(),
            "metrics": {
                "attribute_metrics":
                    {"SEP": [],
                    "LIR": [],
                    "ETD": [],
                    "TID": [],
                    "TPD": [],
                    "MID": [],
                    "Overlap-Attributes": [],
                    "Overlap-Items": [],
                    "Path-Misses": []},
                "items_cluster_metrics":
                    {"Mean Items Per Cluster": [],
                    "Std Items Per Cluster": [],
                    "Clusters Entropy": []},
                "cluster_metrics":{
                    "Silhouette": [],
                    "Calinski Harabasz Index": [],
                    "Davies Bouldin Index:": []
                }
            }
        }

        all_user_ret = {}
        users = self.dataset.get_users('test')
        if verbose: print(f'''Explanation Algorithm {self.model_name}\n''')

        if self.n_users != 0:
            users = users[:self.n_users]

        for user_id in users:
            expl_obj = self.user_explanation(user=user_id, remove_seen=remove_seen, verbose=verbose,
                                             show_dendrogram=False)
            all_user_ret[user_id] = expl_obj
            ret_obj["grid_items"] = pd.concat([ret_obj["grid_items"].copy(),
                                               expl_obj["grid_items"]], ignore_index=True)

            for key in ret_obj["metrics"].keys():
                for key1, value1 in expl_obj['metrics'][key].items():
                    if value1 != -1:
                        ret_obj['metrics'][key][key1].append(value1)

        # all metrics are their mean excluding TID, TPD and Misses
        ret_obj["top_k"] = self.top_k
        for key in ret_obj["metrics"].keys():
            for key1, value_list in ret_obj['metrics'][key].items():
                if key1 != "TPD" and key1 != "TID" and not("Misses" in key1):
                    ret_obj['metrics'][key][key1] = np.array(value_list).mean()
                else:
                    if "Misses" in key1:
                        ret_obj['metrics'][key][key1] = np.array(value_list).sum()
                    else:
                        ret_obj['metrics'][key][key1] = len({item for sublist in value_list for item in sublist})

        return ret_obj, all_user_ret
