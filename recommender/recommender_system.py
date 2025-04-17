import inspect
import json
import os

import cornac.models
import pandas as pd

from dataset_experiment.dataset_experiment import DatasetExperiment
from dataset_experiment import metrics
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.cornac.cornac_utils import predict_ranking


class RecommenderSystem:
    def __init__(self, model: cornac.models.Recommender, dataset: DatasetExperiment,
                 load_path=None, remove_seen=True):
        """
        Recommendation class. It generates a list of recommendations for the user and also extract
        metrics. The class uses cornac recommender engines and evaluates with the recommenders library
        :param model: cornac model to run
        :param dataset: dataset experiment to run the model on
        :param remove_seen: True to remove seen items in the recommendation list, False to keep them.
        """
        sig = inspect.signature(model.__init__)
        model_params = {param: getattr(model, param) for param in sig.parameters
                        if hasattr(model, param)}
        model_params = {k: v for k, v in model_params.items() if not isinstance(v, dict)}

        name_params = model_params["name"] + "&"
        del model_params["name"]
        for key in model_params.keys():
            name_params = name_params + str(key) + "=" + str(model_params[key]) + "&"

        self.model_name = name_params[:-1]
        self.dataset = dataset
        self.remove_seen = remove_seen
        self.load_path = load_path
        self.model = model

    def fit_model(self, save=True) -> object:
        """
        Function to fit the model into the training dataset
        :param save: if true, save trained model on file system. This parameter is ignored if load_path on class
        constructor is not None
        :return: fitted model
        """
        if self.load_path is None:
            print("Training model...")
        else:
            path = self.get_path("model") + self.load_path
            print("Loading model from " + path + " ...")
            self.model = type(self.model).load(path)
            self.model.train_set = self.dataset.train
            return self.model

        if self.dataset.validation is not None:
            self.model.fit(train_set=self.dataset.train, val_set=self.dataset.validation)
        else:
            self.model.fit(train_set=self.dataset.train)

        if save:
            path = self.get_path("model")
            if not (os.path.exists(path)):
                os.mkdir(path)
            self.model.save(save_dir=path, save_trainset=True)

        return self.model

    def recommend_to_user(self, user_id: str, k: int) -> list:
        """
        Function to recommend
        :param user_id: id of the user to generate recommendations to
        :param k: number of recommendations to generate
        :return: list of recommendations items ids in the cornac dataset
        """
        return self.model.recommend(user_id=user_id, k=k,
                                    train_set=self.dataset.train,
                                    remove_seen=self.remove_seen)

    def run_experiment(self, k_list: list, expl_results: dict, rows=3, cols=2, save_results=False, verbose=True) -> dict:
        """
        Run experiment where the recommender system will score all items to all users and extract results.
        This function can also save the score of items for all user, item tuples and the metrics in the file
        system if the flag save_results is set to True.
        :param k_list: list of top k to evaluate
        :param expl_results: dictionary with explanation results from the explanation algorithms
        :param rows: number of rows on grid to evaluate ndcg-2d
        :param cols: number of columns on grid to evaluate ndcg-2d
        :param save_results: True if results should be saved in the datasets folder as file, False otherwise
        :param verbose: True to display results on the console
        :return: metrics as dictionary and saved files on file system
        """
        print("Generating Predictions...")
        train_df, _, test_df = self.dataset.load_fold_asdf()
        predictions = predict_ranking(self.model, train_df,
                        usercol=self.dataset.user_column, itemcol=self.dataset.item_column,
                        predcol=self.dataset.rating_column, remove_seen=self.remove_seen)

        if save_results:
            path = self.get_path("outputs")
            path = path + self.model_name + ".csv"
            predictions.to_csv(path, header=True, index=False, mode="w+")

        print("Generating Ranking Metrics...")
        metrics_value = self.__evaluate(predictions=predictions, test_recs=test_df, k_list=k_list, verbose=verbose)

        for k in k_list:
            metrics_value[f'''Algorithm {self.model.name} NDCG - 2D@{k}'''] = \
                (metrics.ndcg_2d(predictions=predictions, grid_predictions=None, test_recs=test_df,
                                k=k, alg_name=self.model.name, col_rating=self.dataset.rating_column,
                                col_user=self.dataset.user_column, col_item=self.dataset.item_column,
                                alpha=1, beta=1, gama=1, rows=rows, columns=cols, step_x=1,
                                step_y=1, verbose=verbose))

            for key in expl_results.keys():
                try:
                    grid_predictions = expl_results[key]["grid_items"]
                except KeyError:
                    print(f'''Error model {key} does not outputted the grid predictions''')
                    continue

                metrics_value[f'''Algorithm {key} NDCG - 2D@{k}'''] = metrics.ndcg_2d(predictions=predictions,
                                                        grid_predictions=grid_predictions, test_recs=test_df,
                                                        k=k, alg_name=key, col_rating=self.dataset.rating_column,
                                                        col_user='userId', col_item='movieId', alpha=1, beta=1,
                                                        gama=1, rows=rows, columns=cols, step_x=1, step_y=1,
                                                        verbose=verbose)

        explanation_algorithms = []
        for key, value in expl_results.items():
            explanation_algorithms.append({"name": key, "explanation_metrics": value["metrics"]})
        metrics_value["explanation_algoritms"] = explanation_algorithms

        if save_results:
            path = self.get_path("results")
            path = path + self.model_name + ".txt"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(metrics_value, f, indent=4)

        return metrics_value

    def __evaluate(self, predictions: pd.DataFrame, test_recs: pd.DataFrame, k_list: list, verbose=True) -> dict:
        """
        Function that evaluate the predictions generated by the recommender system on top k items
        :param predictions: dataframe containing scores for all possible (user item) tuples, therefore, it has
        three columns: user, item and predicted rating
        :param test_recs: testing set as pandas dataframe
        :param k_list: list of top k recommendations to extract metrics
        :param verbose: True to print the results on console, False otherwise
        :return: metrics as dictionary
        """
        metrics_dict = {}

        for k in k_list:
            eval_map = map(test_recs, predictions, col_user='userId',
                           col_item='movieId',
                           col_prediction=self.dataset.rating_column, k=k)
            eval_ndcg = ndcg_at_k(test_recs, predictions, col_user='userId',
                                  col_item='movieId',
                                  col_prediction=self.dataset.rating_column, k=k)
            eval_precision = precision_at_k(test_recs, predictions, col_user='userId',
                                            col_item='movieId', col_prediction=self.dataset.rating_column, k=k)
            eval_recall = recall_at_k(test_recs, predictions, col_user='userId',
                                      col_item='movieId', col_prediction=self.dataset.rating_column, k=k)

            metrics_dict[f'''MAP@{k}'''] = eval_map
            metrics_dict[f'''NDCG@{k}'''] = eval_ndcg
            metrics_dict[f'''Precision@{k}'''] = eval_precision
            metrics_dict[f'''Recall@{k}'''] = eval_recall

            if verbose:
                print(f'''--- Metrics ---''')
                print(f'''MAP@{k}: {eval_map}''')
                print(f'''NDCG@{k}: {eval_ndcg}''')
                print(f'''Precision@{k}: {eval_precision}''')
                print(f'''Recall@{k}: {eval_recall}''')

        return metrics_dict

    def get_path(self, last_folder: str) -> str:
        """
        Get path to save a file or folder
        :param last_folder: it will be outputs or model or results to save the file in the appropriate folder
        :return: path in the file system as str
        """
        path = self.dataset.path
        if self.dataset.fold_loaded == -1:
            path = path + f'''/stratified_split/{last_folder}/'''
        else:
            path = path + f'''/folds/{self.dataset.fold_loaded}/{last_folder}/'''

        return path
