import os

import cornac.models

from dataset_experiment.dataset_experiment import DatasetExperiment


class ExplanationAlgorithm:
    def __init__(self, dataset: DatasetExperiment, model: cornac.models.Recommender, expr_file: str,
                 top_k: int, n_users=0):
        """
        Explanation algorithm class constructor. Every explanation algorithm uses a kg and the train set
        to have access to the users profile items. The model is required to get the recommendations for the explanations
        :param dataset: dataset used in the recommendation model
        :param model: cornac model used to generate recommendations
        :param expr_file: name of the experiment file configuration
        :param top_k: number of recommendations to explain
        :param n_users: number of users to generate explanations to. If 0 runs to all users
        """
        self.model_name = "ExplanationAlgorithm"
        self.dataset = dataset
        self.model = model
        self.top_k = top_k
        self.n_users = n_users

        path = self.dataset.path
        if self.dataset.fold_loaded == -1:
            self.expl_file_path = path + f'''/stratified_split/explanations/{expr_file[:-5]}/'''
        else:
            self.expl_file_path = path + f'''/folds/{self.dataset.fold_loaded}/explanations/{expr_file[:-5]}/'''

        try:
            os.makedirs(self.expl_file_path, exist_ok=True)
        except FileExistsError:
            pass

        self.memo_sep = {}

    def user_explanation(self, user: str, remove_seen=True, verbose=True, **kwargs) -> dict:
        """
        Function to generate
        :param user: user id to show explanations to
        :param remove_seen: True if model should exclude seen items, False otherwise
        :param verbose: True to print explanations
        :return: dict where key is recommended item and value is explanation
        """
        pass

    def all_users_explanations(self, remove_seen=True, verbose=True) -> tuple[dict, dict]:
        """
        Method to run explanations to all users and extract explanation metrics
        :param remove_seen: remove seen items on evaluation
        :param verbose: True to display log, False otherwise
        :return: tuple of two dictionaries: one containing the metrics and the other one with all outputs of all users.
        """
        pass