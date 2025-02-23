import os
import pandas as pd

from datasets.dataset_experiment import DatasetExperiment
from sklearn.model_selection import KFold
from recommenders.datasets.python_splitters import python_stratified_split
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

class MovieLens100K(DatasetExperiment):

    __K_FOLDS = 5

    def __init__(self, gen_dataset=False):
        """
        MovieLens 100k class constructor
        :param gen_dataset: if true generate the dataset on class constructor
        """
        super().__init__(name="MovieLens100k", path="./datasets/ml-latest-small",
                         user_column="userId", item_column="movieId", rating_column="rating",
                         original_file_name="ratings_processed.csv", k_folds=self.__K_FOLDS,
                         split_percentage=0.8, gen_dataset=gen_dataset)

        self.prop_set = pd.read_csv("./knowledge-graphs/props_wikidata_movielens_small.csv",
                                    usecols=['movieId', 'title', 'prop', 'obj'])
        self.prop_set = self.prop_set.set_index('movieId')

        if gen_dataset:
            self.generate_dataset()

    def generate_dataset(self):
        """
        Function to generate the dataset  of the MovieLens 100k dataset.
        There are two main folders: the stratified split according to users, to keep the same proportion
        on training and test sets and the K-Fold which is made by initially creating a train and a test
        set for the __K_FOLDS folds then a stratified split is made on the test to create the validation set
        :return: K-folds created on the dataset folder with its number as the number of the folder
        """
        original_dataset = self.load_original()

        # creating stratified split
        path = "./datasets/ml-latest-small/stratified_split/"
        if not (os.path.exists(path)):
            os.mkdir(path)
            os.mkdir(path + "/outputs")
            os.mkdir(path + "/results")

        train_s, test_s = python_stratified_split(original_dataset, self.split_percentage,
                                                  col_user='userId', col_item='movieId')
        original_dataset = original_dataset.drop(columns=["random"])

        equal = not (pd.concat([train_s, test_s])
            .sort_values(["userId", "movieId", "timestamp", "rating"])
            .reset_index(drop=True)
            .compare(original_dataset.sort_values(["userId", "movieId", "timestamp", "rating"])).shape[0])

        if equal:
            train_s.to_csv(path + "/train.csv", header=True, index=False)
            test_s.to_csv(path + "/test.csv", header=True, index=False)
        else:
            raise Exception("Ops! Concatenation of training, validation and test sets do "
                            "not result in the original dataset!")

        # creating k_fold
        k_folds = KFold(n_splits=self.__K_FOLDS, shuffle=True, random_state=42)
        folds = list(k_folds.split(original_dataset))
        s_path = "./datasets/ml-latest-small/folds"
        if not (os.path.exists(s_path)):
            os.mkdir(s_path)
            os.mkdir(s_path + "/outputs")
            os.mkdir(s_path + "/results")

        for f in range(0, self.__K_FOLDS):
            fold_name = "./datasets/ml-latest-small/folds/" + str(f)
            if not (os.path.exists(fold_name)):
                os.mkdir(fold_name)
                os.mkdir(fold_name + "/outputs")
                os.mkdir(fold_name + "/results")

            train = original_dataset.iloc[folds[f][0]]
            val_test = original_dataset.iloc[folds[f][1]]

            validation, test = python_stratified_split(val_test, ratio=0.5,
                                                       col_user='userId', col_item='movieId',
                                                       seed=self.seed)
            val_test.drop(columns=["random"], inplace=True)

            equal = not (pd.concat([train, validation, test])
                .sort_values(["userId", "movieId", "timestamp", "rating"])
                .reset_index(drop=True)
                .compare(original_dataset.sort_values(["userId", "movieId", "timestamp", "rating"])).shape[0])

            if equal:
                train.to_csv(fold_name + "/train.csv", header=True, index=False)
                validation.to_csv(fold_name + "/validation.csv", header=True, index=False)
                test.to_csv(fold_name + "/test.csv", header=True, index=False)
            else:
                raise Exception("Ops! Concatenation of training, validation and test sets do "
                                "not result in the original dataset!")
