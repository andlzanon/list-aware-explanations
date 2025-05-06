import os

import cornac.models
import numpy as np
import pandas as pd

from dataset_experiment import metrics
from dataset_experiment.dataset_experiment import DatasetExperiment
from explanations.explanation import ExplanationAlgorithm


class ExpLOD(ExplanationAlgorithm):
    def __init__(self, dataset: DatasetExperiment, model: cornac.models.Recommender, top_k: int,
                 top_n=1, hitems_per_attr=2, n_users=0):
        """
        :param dataset:
        :param model:
        :param top_k: top k items to explain
        :param top_n: number of top attributes to generate the explanation
        :param hitems_per_attr: number of historic items showed per attribute on explanation.
            In an example such as: I recommend you Titanic since you ofter like drama items as X, Y, Z.
            hitems_per_attr is 3, because we are using X, Y and Z profile items to support the attribute
        :param n_users: number of users to generate explanations to. If 0 runs to all users
        """
        super().__init__(dataset, model, top_k, n_users)
        self.top_n = top_n
        self.hitems_per_attr = hitems_per_attr
        self.model_name = (f"ExpLOD&top_n={str(self.top_n)}&hitems_per_attr={str(self.hitems_per_attr)}"
                           f"&top_k={str(self.top_k)}&u={str(abs(self.n_users))}")
        self.expl_file_path = self.expl_file_path + self.model_name + ".txt"
        open(self.expl_file_path, 'w+', encoding='utf-8').close()

    def __user_semantic_profile(self, historic: list) -> dict:
        """
        Generate the user semantic profile, where all the values of properties (e.g.: George Lucas, action films, etc.)
        are ordered by a score that is calculated as:
            score = (npi/i) * log(N/dft)
        where npi are the number of edges to a value, i the number of interacted items,
        N the total number of items and dft the number of items with the value
        :param historic: list of the items interacted by a user
        :return: dictionary with properties' values as keys and scores as values
        """

        # create npi, i and n columns
        interacted_props = self.dataset.prop_set.loc[self.dataset.prop_set.index.isin(historic)].copy()
        interacted_props['npi'] = interacted_props.groupby(self.dataset.prop_set.columns[-1])[
            self.dataset.prop_set.columns[-1]].transform('count')
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

        items_historic = [next((int(k) for k, v in self.dataset.train.iid_map.items() if v == u_item), None)
                          for u_item in self.dataset.train.chrono_user_data[self.dataset.train.uid_map[user]][0]]
        ranked_items = list(self.model.recommend(user_id=user, k=self.top_k,
                                                 train_set=self.dataset.train,
                                                 remove_seen=remove_seen))
        
        train_set = self.dataset.load_fold_asdf()[0]
        semantic_profile = self.__user_semantic_profile(items_historic)

        # get properties from historic and recommended items
        hist_props = self.dataset.prop_set.loc[items_historic]
        prop_cols = self.dataset.prop_set.columns

        with open(self.expl_file_path, 'a+', encoding='utf-8') as f:
            f.write(f'''--- Explanations User Id {user} ---\n''')
        if verbose: print(f'''--- Explanations User Id {user} ---''')
        for r in ranked_items:
            rec_props = self.dataset.prop_set.loc[int(r)]

            # check properties on both sets
            intersection = pd.Series(list(set(hist_props['obj']).intersection(set(rec_props['obj']))))

            # generate dictionary only of attributes on recommended item
            r_sem_dict = {}
            for p in intersection:
                r_sem_dict[p] = semantic_profile[p]

            props_sorted = sorted(r_sem_dict.items(), key=lambda item: item[1], reverse=True)
            max_props = [k for k, _ in props_sorted[:self.top_n]]

            # build sentence
            train_c = train_set.copy()
            train_c = train_c.set_index(train_set.columns[0])
            user_df = train_c.loc[str(user)]
            try:
                rec_name = self.dataset.prop_set.loc[int(r)]['title'].unique()[0]
            except AttributeError:
                rec_name = self.dataset.prop_set.loc[int(r)][prop_cols[0]]
            full_sentence = "I recommend you " + "\"" + str(rec_name) + "\" since you often like "

            for pi in range(0, len(max_props)):
                p = max_props[pi]
                path_sentence = "\"" + p + "\" items as "
                user_item = user_df[
                    user_df[user_df.columns[0]].isin(
                        list(hist_props[hist_props['obj'] == p].index.unique().astype(str)))]
                hist_ids = list(
                    user_item.sort_values(by=user_item.columns[-1],
                                          ascending=False)[:self.hitems_per_attr][user_item.columns[0]].astype(int))
                hist_names = hist_props.loc[hist_ids][prop_cols[0]].unique()

                # check for others with same value
                hist_sentence = ""
                for i in range(0, len(hist_names)):
                    hname = hist_names[i]
                    hist_sentence += "\"" + hname + "\", "
                hist_sentence_parts = hist_sentence[:-2].rsplit(", ", 1)
                hist_sentence = " and ".join(hist_sentence_parts) if len(hist_sentence_parts) > 1 else hist_sentence
                path_sentence += hist_sentence

                full_sentence += path_sentence
                if self.top_n > 1 and pi <= self.top_n-1:
                    if pi == 0:
                        full_sentence += ". Moreover, I recommend it because you sometimes like "
                    else:
                        full_sentence += " and "

                interacted_items.append(hist_ids)
                attributes.append(max_props)

            user_explanations[int(r)] = full_sentence[:-5]

            with open(self.expl_file_path, 'a+', encoding='utf-8') as f:
                f.write("Recommended Item: " + str(r) + ": " + str(rec_name) + "\n")
                f.write(full_sentence + "\n\n")
            if verbose:
                print("Recommended Item: " + str(r) + ": " + str(rec_name))
                print(full_sentence + "\n")

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
        overlap_items = len(unique_items) / total_items

        expl_metrics = {
            "attribute_metrics": {
                "SEP": sep,
                "LIR": lir,
                "ETD": etd,
                "TID": unique_items,
                "TPD": unique_attributes,
                "MID": mid,
                "Overlap-Attributes": overlap_attributes,
                "Overlap-Items": overlap_items
            }
        }

        ret_obj = {
            "explanations": user_explanations,
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
            "metrics": {
                "attribute_metrics":
                    {"SEP": [],
                     "LIR": [],
                     "ETD": [],
                     "TID": [],
                     "TPD": [],
                     "MID": [],
                     "Overlap-Attributes": [],
                     "Overlap-Items": []},
            }
        }

        all_user_ret = {}
        users = self.dataset.get_users('test')
        if self.n_users != 0:
            users = users[:self.n_users]

        if verbose: print(f'''Explanation Algorithm {self.model_name}\n''')
        for user_id in users:
            expl_obj = self.user_explanation(user=user_id, remove_seen=remove_seen,  verbose=verbose)
            all_user_ret[user_id] = expl_obj

            for key in ret_obj["metrics"].keys():
                for key1, value1 in expl_obj['metrics'][key].items():
                    ret_obj['metrics'][key][key1].append(value1)

        for key in ret_obj["metrics"].keys():
            for key1, value_list in ret_obj['metrics'][key].items():
                if key1 != "TPD" and key1 != "TID":
                    ret_obj['metrics'][key][key1] = np.array(value_list).mean()
                else:
                    ret_obj['metrics'][key][key1] = len({item for sublist in value_list for item in sublist})

        return ret_obj, all_user_ret
        