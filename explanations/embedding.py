import itertools
import math
import os

import cornac
import numpy as np
import pandas as pd
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import NearestNeighbors

from dataset_experiment import metrics
from dataset_experiment.dataset_experiment import DatasetExperiment
from explanations.explanation import ExplanationAlgorithm


class EmbeddingClustering(ExplanationAlgorithm):

    def __init__(self, dataset: DatasetExperiment, alg: str, model: cornac.models.Recommender, expr_file: str,
                 top_k: int, n_users: int, emb_model_path: str, kg_emb_model_name: str, n_clusters: int,
                 top_n: int, hitems_per_attr=2, random_state=42, normalize=True,
                 local_weight=0.5, global_weight=0.5, **kwargs):
        """
        Class to generate explanations based on KG embeddings to cluster items.
        :param dataset: dataset used in the recommendation model
        :param alg: name of the clustering dataset
        :param model: cornac model used to generate recommendations
        :param expr_file: name of the experiment file configuration
        :param top_k: number of recommendations to explain
        :param n_users: number of users to generate explanations to. If 0 runs to all users
        :param emb_model_path: path for the KG embedding model. If there is no model in the path,
            train one based on emb_model_name
        :param emb_model_name: name of the KG embedding model
        :param n_clusters: number of clusters
        :param top_n: number of attributes to shown in explanation
        :param hitems_per_attr: number of historic items showed per attribute on explanation.
        :param random_state: random state
        :param normalize: true to locally normalize KG embedding that have magnitude
        :param kwargs: arguments for the KG embedding model if it does not exist on the emb_model_path
        """
        super().__init__(dataset, model, expr_file, top_k, n_users)
        self.alg = alg
        self.emb_model_path = emb_model_path
        self.random_state = random_state
        self.normalize = normalize
        self.n_clusters = n_clusters
        self.top_n = top_n
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.hitems_per_attr = hitems_per_attr
        self.kg_emb_model_name = kg_emb_model_name

        self.kg_embed_params = kwargs.get("embedding_model_params")
        self.kg_tf = TriplesFactory.from_labeled_triples(
            triples=self.dataset.prop_set.reset_index()[[self.dataset.item_column, 'prop', 'obj']].astype(str).values)
        self.kg_embed_train, self.kg_embed_val, self.kg_embed_test = self.kg_tf.split([0.8, 0.1, 0.1],
                                                                                      random_state=self.random_state)
        self.global_edge_rel, self.global_attr_rel = self.__compute_global_score()

        try:
            self.emd_model = torch.load(self.emb_model_path + "/trained_model.pkl")
        except Exception:
            self.__generate_emb_model(kg_emb_model_name, self.random_state,
                                      self.emb_model_path, self.kg_embed_params)
            self.emd_model = torch.load(self.emb_model_path + "trained_model.pkl")

        self.model_name = (f"Embed&alg={str(self.alg)}&method={str(self.kg_emb_model_name)}"
                           f"&n_clusters={str(self.n_clusters)}&top_n={str(self.top_n)}"
                           f"&hitems_per_attr={str(self.hitems_per_attr)}"
                           f"&rs={str(self.random_state)}&top_k={str(self.top_k)}&lw={str(self.local_weight)}"
                           f"&gw={str(self.global_weight)}&u={str(abs(self.n_users))}"
                           f"&normalize={str(self.normalize)}")
        self.expl_file_path = r"\\?\\" + os.path.abspath(self.expl_file_path + self.model_name + ".txt")
        open(self.expl_file_path, 'w+').close()

    def __compute_global_score(self):
        """
        Generate a global popularity score of edges (called in this code as prop) and attributes.
        It is the normalized value of the quantity of items an attribute is connected to.
        For edges, it is the normalized value of the frequency of the edge across items.
        :return: two dictionaries: one for the attribute score and another to the edges scores
        """
        prop_col = self.dataset.prop_set.columns[-2]
        obj_col = self.dataset.prop_set.columns[-1]

        item_kg = self.dataset.prop_set.copy()
        item_kg['obj_count'] = item_kg.groupby(obj_col)[prop_col].transform(
            lambda x: x.index.nunique())
        item_kg['global_count_norm'] = (
                (item_kg["obj_count"] - item_kg["obj_count"].min()) / (item_kg["obj_count"].max() - item_kg["obj_count"].min()))

        item_kg['prop_count'] = item_kg.groupby([prop_col])[
            item_kg.columns[-1]].transform(
            'count')
        item_kg['prop_count_norm'] = (item_kg["prop_count"] - item_kg["prop_count"].min()) / (
                    item_kg["prop_count"].max() - item_kg["prop_count"].min())

        obj_dict = item_kg[[obj_col, 'global_count_norm']].drop_duplicates().\
            sort_values(by='global_count_norm', ascending=False).\
            set_index(obj_col).\
            to_dict()['global_count_norm']

        prop_dict = item_kg[[prop_col, 'prop_count_norm']].drop_duplicates().\
            sort_values(by='prop_count_norm',ascending=False).\
            set_index(prop_col).to_dict()['prop_count_norm']

        return prop_dict, obj_dict

    def __user_semantic_profile(self, historic: list) -> tuple[dict, dict]:
        """
        Generate the user semantic profile, where all the values of properties (e.g.: George Lucas, action films, etc.)
        are ordered by a score that is calculated as:
            score = (npi/i) * log(N/dft)
        where npi are the number of edges to a value, i the number of interacted items,
        N the total number of items and dft the number of items with the value
        :param historic: list of the items interacted by a user
        :return: dictionary where pos 0 is properties' values as keys and scores as values and 1 the frequency of
            attributes on interacted items
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
        top_prop = interacted_props['npi'].to_dict()
        fav_prop = interacted_props['score'].to_dict()

        return fav_prop, top_prop

    def __generate_emb_model(self, emb_model_name: str, seed, dir, model_params: dict):
        train, val, test = self.kg_embed_train, self.kg_embed_val, self.kg_embed_test

        pipeline_result = pipeline(
            training=train,
            validation=val,
            testing=test,
            model=emb_model_name,
            model_kwargs=dict(
                embedding_dim=model_params["embedding_dim"],
            ),
            optimizer=model_params["optimizer"],
            optimizer_kwargs=dict(lr=model_params["lr"]),
            loss=model_params["loss"],
            random_seed=seed,
            device=model_params["device"],
            negative_sampler=model_params["negative_sampler"],
            negative_sampler_kwargs=dict(num_negs_per_pos=model_params["num_negs_per_pos"], filtered=True),
            stopper='early',
            use_testing_data=False,
            stopper_kwargs=dict(frequency=5, patience=3, relative_delta=0.002,
                                metric='hits@5', larger_is_better=True),
            training_loop=model_params["training_loop"],
            training_kwargs=dict(num_epochs=model_params["num_epochs"], batch_size=model_params["batch_size"]),
        )

        print(f"Hits@1: {pipeline_result.get_metric('hits@1')}")
        print(f"Hits@3: {pipeline_result.get_metric('hits@3')}")
        print(f"Hits@5: {pipeline_result.get_metric('hits@5')}")
        print(f"Hits@10: {pipeline_result.get_metric('hits@10')}")
        print(f"Mean Reciprocal Rank: {pipeline_result.get_metric('mean_reciprocal_rank')}")
        pipeline_result.save_to_directory(dir)

        return torch.load(dir + "/trained_model.pkl")

    def user_explanation(self, user: str, remove_seen=True, verbose=True, **kwargs) -> dict:
        """
        Generate user explanation based on KE embeddings
        :param user: user id
        :param remove_seen: True if model should exclude seen items, False otherwise
        :param verbose: True to print explanations
        :param kwargs: additional arguments
        :return: user explanation metrics
        # TODO add edge embedding
        # TODO: optimize number of clusters
        """
        user_explanations = {}
        interacted_items = []
        attributes = []
        ranked_clusters = []
        rem_items = []
        path_misses = 0
        cluster_misses = 0
        name_col = self.dataset.prop_set.columns[0]
        obj_col = self.dataset.prop_set.columns[-1]

        with open(self.expl_file_path, 'a+', encoding='utf-8') as f:
            f.write(f'''--- Explanations User Id {user} ---\n''')
        if verbose: print(f'''--- Explanations User Id {user} ---''')

        # generate user recommendations
        ranked_items = list(self.model.recommend(user_id=user, k=self.top_k,
                                                 train_set=self.dataset.train,
                                                 remove_seen=remove_seen))

        # get user historic items
        items_historic = [next((int(k) for k, v in self.dataset.train.iid_map.items() if v == u_item), None)
                          for u_item in
                          self.dataset.train.user_data[self.dataset.train.uid_map[user]][0]]

        # get relevance of attributes locally, based on the interacted items
        local_attr_rel = self.__user_semantic_profile(items_historic)[0]
        # get the entity embeddings
        entity_embeds = pd.DataFrame(
            self.emd_model.entity_representations[0](indices=None).detach().cpu().numpy()).rename(
            self.kg_tf.entity_id_to_label)

        # get attributes of user interacted items, recommended items and generate set of union
        pro_all_attr = self.dataset.prop_set.loc[items_historic][obj_col]
        rec_all_attr = self.dataset.prop_set.loc[list(map(int, ranked_items))][obj_col]
        inter = set(rec_all_attr).intersection(set(pro_all_attr))
        inter = np.array(sorted(inter))

        # initializing clustering dataframe
        clustering_df = pd.DataFrame(columns=range(self.kg_embed_params["embedding_dim"]))

        # generating item embedding as a weighted sum of its attribute embeddings
        co_ocur_df = pd.DataFrame(columns=inter)
        for rec_item in ranked_items:
            rec_attr = self.dataset.prop_set.loc[int(rec_item)][obj_col]
            vi_num = 0
            vi_den = 0

            # create the vector array (vectorize) of a recommended item based on binary presence of attributes
            # for explanations
            # TODO: check path misses and cluster misses
            vectorize = np.isin(inter, rec_attr).astype(int)
            if sum(vectorize) > 0:
                co_ocur_df.loc[len(co_ocur_df)] = vectorize
            else:
                rem_items.append(rec_item)
                path_misses = path_misses + 1
                continue

            # create weighted embeddings for clustering algorithm
            for attr in rec_attr:
                attr_embed = entity_embeds.loc[attr].to_numpy()
                attr_embed = attr_embed / np.linalg.norm(attr_embed)
                try:
                    local_rel = local_attr_rel[attr]
                except KeyError:
                    continue

                global_rel = self.global_attr_rel[attr]
                total_weight =  (self.local_weight * local_rel) + (self.global_weight * global_rel)
                vi_num = vi_num + (total_weight * attr_embed)
                vi_den = vi_den + total_weight

            vi = vi_num/vi_den
            clustering_df.loc[len(clustering_df)] = vi

        # run clustering
        clustering_data = clustering_df.to_numpy()
        linkage_matrix = linkage(clustering_data, method="weighted", metric="cosine")
        clusters = fcluster(linkage_matrix, t=self.n_clusters, criterion="maxclust")

        n_clusters = 0
        for i in range(min(clusters), max(clusters) + 1):
            n_clusters = n_clusters + 1
            # get items on cluster, then the attributes of the items on the cluster
            i_cluster = [j for j in range(0, len(clusters)) if clusters[j] == i]

            # get the top n attributes common across all items in the cluster
            cluster_attr = co_ocur_df.iloc[i_cluster]
            cluster_sum = cluster_attr.sum(axis=0)
            expl_attr_names = cluster_sum[cluster_sum == len(i_cluster)].sort_index().index.tolist()
            n_attr = len(expl_attr_names)

            # if there is no expl_attr_names where there are common attributes
            if n_attr == 0:
                cluster_misses = cluster_misses + 1
                continue

            if n_attr == 1:
                expl_attr_names = [expl_attr_names[0]]
            else:
                attribute_data = pd.DataFrame(columns=range(self.kg_embed_params["embedding_dim"]))
                for attr in expl_attr_names:
                    attribute_data.loc[len(attribute_data)] = entity_embeds.loc[attr]

                attribute_data = attribute_data.to_numpy()
                nn = NearestNeighbors(n_neighbors=min(self.top_n, n_attr), metric="cosine")
                try:
                    nn.fit(attribute_data)
                    distances, indices = nn.kneighbors(attribute_data)
                except ValueError:
                    nn.fit(attribute_data.real)
                    distances, indices = nn.kneighbors(attribute_data.real)

                best_score = np.inf
                best_group = None
                for row, idxs in enumerate(indices):
                    group_distances = distances[row]
                    score = group_distances.mean()
                    if score < best_score:  # closer = better
                        best_score = score
                        best_group = idxs

                expl_attr_names = [expl_attr_names[i] for i in best_group]

            # get recommended item names
            rec_item_ids = np.array(ranked_items)[i_cluster].astype(int)
            rec_item_names = self.dataset.prop_set.loc[rec_item_ids][name_col].unique()

            # get profile item names that have the explanation attributes
            pro_df = self.dataset.prop_set.loc[items_historic]
            pro_item_ids = pro_df.groupby(pro_df.index)[obj_col].apply(set)
            pro_item_ids = pro_item_ids.apply(lambda attrs: set(attrs).issuperset(set(expl_attr_names)))
            pro_item_ids = pro_item_ids[pro_item_ids == True].index.astype(int)
            pro_item_ids = np.random.choice(pro_item_ids,
                                            size=pro_item_ids.shape[0], replace=False)[:self.hitems_per_attr]
            pro_item_names = self.dataset.prop_set.loc[pro_item_ids][name_col].unique().tolist()

            interacted_items.append(pro_item_ids)
            attributes.append(expl_attr_names)

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
            else:
                raise AttributeError("Profile items array and shared attributes array lengths are 0")

            if verbose: print(expl)
            with open(self.expl_file_path, 'a+', encoding='utf-8') as f:
                f.write(expl)
            for rec in rec_item_ids:
                user_explanations[rec] = expl

        try:
            clu_metrics = metrics.clustering_metrics(clustering_data, clusters, verbose=False)
        except ValueError:
            clu_metrics = metrics.clustering_metrics(clustering_data.real, clusters, verbose=False)

        item_cluster_metrics = metrics.items_per_cluster(clusters.tolist())

        unique_items = list(set([item for sublist in interacted_items for item in sublist]))
        unique_attributes = list(set([item for sublist in attributes for item in sublist]))
        total_attributes = sum([len(sublist) for sublist in attributes])
        total_items = sum([len(sublist) for sublist in interacted_items])

        mid = np.array([len(sublist) for sublist in interacted_items]).mean()
        lir = metrics.lir_metric(beta=0.3, user=user, items=unique_items,
                                train_set=self.dataset.load_fold_asdf()[0],
                                col_user=self.dataset.user_column, col_item=self.dataset.item_column)
        sep = metrics.sep_metric(beta=0.3, props=attributes, prop_set=self.dataset.prop_set, memo_sep=self.memo_sep)

        try:
            etd = metrics.etd_metric(unique_attributes, self.top_k, total_attributes)
        except ZeroDivisionError:
            etd = math.nan

        try:
            overlap_attributes = len(unique_attributes) / total_attributes
        except ZeroDivisionError:
            overlap_attributes = math.nan

        try:
            overlap_items = len(unique_items) / total_items
        except ZeroDivisionError:
            overlap_items = math.nan

        cluster_misses = cluster_misses / n_clusters

        attr_metrics = {
            "SEP": sep,
            "LIR": lir,
            "ETD": etd,
            "TID": unique_items,
            "TPD": unique_attributes,
            "MID": mid,
            "Overlap-Attributes": overlap_attributes,
            "Overlap-Items": overlap_items,
            "Cluster-Misses": cluster_misses,
            "Path-Misses": path_misses
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
                    rerank.loc[len(rerank)] = [user, item, x + 1, y + 1]

        expl_metrics = {
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
                     "Cluster-Misses": [],
                     "Path-Misses": []},
                "items_cluster_metrics":
                    {"Mean Items Per Cluster": [],
                     "Std Items Per Cluster": [],
                     "Clusters Entropy": [],
                     "Number of Clusters": []},
                "cluster_metrics": {
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
            expl_obj = self.user_explanation(user=user_id, remove_seen=remove_seen, verbose=verbose)
            all_user_ret[user_id] = expl_obj
            ret_obj["grid_items"] = pd.concat([ret_obj["grid_items"].copy(),
                                               expl_obj["grid_items"]], ignore_index=True)

            for key in ret_obj["metrics"].keys():
                for key1, value1 in expl_obj['metrics'][key].items():
                    if not isinstance(value1, list):
                        if not math.isnan(value1):
                            ret_obj['metrics'][key][key1].append(value1)
                    else:
                        ret_obj['metrics'][key][key1].append(value1)

        # all metrics are their mean excluding TID, TPD and Misses
        ret_obj["top_k"] = self.top_k
        for key in ret_obj["metrics"].keys():
            for key1, value_list in ret_obj['metrics'][key].items():
                if key1 != "TPD" and key1 != "TID" and key1 != "Path-Misses":
                    ret_obj['metrics'][key][key1] = np.array(value_list).mean()
                else:
                    if key1 == "Path-Misses":
                        ret_obj['metrics'][key][key1] = np.array(value_list).sum()
                    else:
                        ret_obj['metrics'][key][key1] = len({item for sublist in value_list for item in sublist})

        return ret_obj, all_user_ret