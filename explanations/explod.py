import cornac.models
import numpy as np
import pandas as pd

from dataset_experiment.dataset_experiment import DatasetExperiment
from explanations.explanation import ExplanationAlgorithm


class ExpLOD(ExplanationAlgorithm):
    def __init__(self, dataset: DatasetExperiment, model: cornac.models.Recommender):
        super().__init__(dataset, model)

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
        
    def user_explanation(self, user: str, top_k: int, remove_seen=True, verbose=True, top_n=1) -> dict:
        """
        Generate user explanation with ExpLOD algorithm link: https://dl.acm.org/doi/abs/10.1145/2959100.2959173
        :param user: user id
        :param top_k: top k items to explain
        :param remove_seen: True if model should exclude seen items, False otherwise
        :param verbose: True to print explanations
        :param top_n: number of top attributes to generate the explanation
        :return: explanations as dict where key is recommended item and value is explanation
        """

        user_explanations = {}
        items_historic = [next((int(k) for k, v in self.dataset.train.iid_map.items() if v == u_item), None)
                          for u_item in self.dataset.train.chrono_user_data[self.dataset.train.uid_map[user]][0]]
        ranked_items = list(self.model.recommend(user_id=user, k=top_k,
                                                 train_set=self.dataset.train,
                                                 remove_seen=remove_seen))
        
        train_set = self.dataset.load_fold_asdf()[0]
        semantic_profile = self.__user_semantic_profile(items_historic)

        # get properties from historic and recommended items
        hist_props = self.dataset.prop_set.loc[items_historic]
        prop_cols = self.dataset.prop_set.columns
        for r in ranked_items:
            rec_props = self.dataset.prop_set.loc[int(r)]

            # check properties on both sets
            intersection = pd.Series(list(set(hist_props['obj']).intersection(set(rec_props['obj']))))

            # generate dictionary only of attributes on recommended item
            r_sem_dict = {}
            for p in intersection:
                r_sem_dict[p] = semantic_profile[p]

            props_sorted = sorted(r_sem_dict.items(), key=lambda item: item[1], reverse=True)
            max_props = [k for k, _ in props_sorted[:top_n]]

            # build sentence
            train_c = train_set.copy()
            train_c = train_c.set_index(train_set.columns[0])

            user_df = train_c.loc[str(user)]
            user_item = user_df[
                user_df[user_df.columns[0]].isin(
                    list(hist_props[hist_props['obj'].isin(max_props)].index.unique().astype(str)))]
            hist_ids = list(
                user_item.sort_values(by=user_item.columns[-1], ascending=False)[:3][user_item.columns[0]].astype(int))
            hist_names = hist_props.loc[hist_ids][prop_cols[0]].unique()
            try:
                rec_name = self.dataset.prop_set.loc[int(r)]['title'].unique()[0]
            except AttributeError:
                rec_name = self.dataset.prop_set.loc[int(r)][prop_cols[0]]

            origin = "Because you watched "
            # check for others with same value
            for i in hist_names:
                origin = origin + "\"" + i + "\"; "
            origin = origin[:-2]

            path_sentence = " that share the attribute "
            for n in max_props:
                path_sentence = path_sentence + "\"" + n + "\" "
            destination = ", watch \"" + rec_name + "\" that has the same attribute"
            full_sentence = origin + path_sentence + destination

            user_explanations[r] = full_sentence
            if verbose:
                print("\nRecommended Item: " + str(r) + ": " + str(rec_name))
                print(full_sentence)

        return user_explanations

    def all_users_explanations(self, top_n: int, output_file: str, remove_seen=True, verbose=True):
        # TODO: implement function
        pass
        