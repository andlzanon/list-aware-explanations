import cornac.models

from dataset_experiment.dataset_experiment import DatasetExperiment


class ExplanationAlgorithm:
    def __init__(self, dataset: DatasetExperiment, model: cornac.models.Recommender):
        """
        Explanation algorithm class constructor. Every explanation algorithm uses a kg and the train set
        to have access to the users profile items. The model is required to get the recommendations for the explanations
        :param dataset: dataset used in the recommendation model
        :param model: cornac model used to generate recommendations
        """
        self.dataset = dataset
        self.model = model
        self.memo_sep = {}

    def user_explanation(self, user: str, top_k: int, remove_seen=True, verbose=True, **kwargs) -> dict:
        """
        Function to generate
        :param user: user id to show explanations to
        :param top_k: number of recommendations to generate explanations to
        :param remove_seen: True if model should exclude seen items, False otherwise
        :param verbose: True to print explanations
        :return: dict where key is recommended item and value is explanation
        """
        pass

    def all_users_explanations(self, top_k: int, output_file: str, remove_seen=True, verbose=True) -> None:
        """
        Function to generate explanations to all users on the train set
        :param top_k: number of recommendations to generate explanations to
        :param output_file: file name to save output
        :param remove_seen: True if model should exclude seen items, False otherwise
        :param verbose: True to print explanations
        :return: File with explanations to all users in file system
        """
        pass