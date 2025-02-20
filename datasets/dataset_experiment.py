import numpy as np
import pandas as pd
from cornac.data import Reader, Dataset

class DatasetExperiment:
    def __init__(self, name: str, path: str, original_file_name: str,
                 user_column: str, item_column: str, rating_column: str, binarize=0,
                 read_format="UIRT", k_folds=5, seed=42):
        """
        Dataset class definition. The Dataset class has three main attributes.
        :param name: name of the dataset
        :param path: absolute path of where the dataset is
        :param original_file_name: name of the file with the original full
        :param user_column: name of the user column on the dataset
        :param item_column: name of the item column on the dataset
        :param rating_column: name of the rating column on the dataset
        :param binarize: if above value transform into 1, else exclude
        :param read_format: can be 'UIRT' for user item rating and timestamp or 'UIR' for user item rating
        :param k_folds: number of folds in which the dataset was divided
        :param seed: seed number
        """
        self.__name = name
        self.__path = path
        self.__original_file_name = original_file_name
        self.user_column = user_column
        self.item_column = item_column
        self.rating_column = rating_column
        self.k_folds = k_folds
        self.folds_set = []
        self.fold_loaded = -1
        self.train = None
        self.validation = None
        self.test = None
        self.binarize = binarize
        self.read_format = read_format
        self.seed = seed

    def load_original(self) -> pd.DataFrame:
        """
        Load the original dataset without preprocessing, not divided in train/validation/test
        :return: dataframe with the original dataset
        """
        return pd.read_csv(self.__path + '''/''' + self.__original_file_name)

    def load_fold(self, i: int) -> tuple[Dataset, Dataset, Dataset]:
        """
        Function to load a fold of the dataset
        :param i: number of the fold to load
        :return: train, validation and test sets as a tuple of DataFrames
        """
        assert(i <= self.k_folds), f'''The dataset {self.__name} has five folds (0 to {self.k_folds-1})'''
        assert self.read_format in ["UIRT", "UIR"], f'''Format should be UIRT or UIR'''

        reader = Reader(bin_threshold=self.binarize)
        path = self.__path + f'''/folds/{i}/'''

        if self.read_format == "UIRT":
            self.train = Dataset.from_uirt(
                reader.read(path + '''train.csv''', sep=',', skip_lines=1, fmt=self.read_format),
                seed=self.seed)
            self.validation = Dataset.from_uirt(
                reader.read(path + '''validation.csv''', sep=',', skip_lines=1, fmt=self.read_format),
                seed=self.seed)
            self.test = Dataset.from_uirt(
                reader.read(path + '''test.csv''', sep=',', skip_lines=1, fmt=self.read_format),
                seed=self.seed)
        else:
            self.train = Dataset.from_uir(
                reader.read(path + '''train.csv''', sep=',', skip_lines=1, fmt=self.read_format),
                seed=self.seed)
            self.validation = Dataset.from_uir(
                reader.read(path + '''validation.csv''', sep=',', skip_lines=1, fmt=self.read_format),
                seed=self.seed)
            self.test = Dataset.from_uir(
                reader.read(path + '''test.csv''', sep=',', skip_lines=1, fmt=self.read_format),
                seed=self.seed)

        self.fold_loaded = i
        return self.train, self.validation, self.test

    def load_all_folds(self) -> list:
        """
        Function that loads all the folds in a list. Each position of the list represent a fold.
        In each position of the list there is a tuple containing a training set, a validation set and a test set
        :return: list with train, validation and test tuples
        """
        self.folds_set = []
        for i in range(0, self.k_folds):
            self.folds_set.append(self.load_fold(i))
        return self.folds_set

    def fold_percentage(self, fold_number=-1) -> None:
        """
        Show fold dataset statistics
        :param fold_number: number of the folder to get statistics, if -1 then use the loaded train, validation
            and test sets on the class attributes
        :return: info about train, validation and test sets
        """

        if fold_number == -1:
            assert ((self.train.num_ratings > 0) & (self.validation.num_ratings > 0) & (self.test.num_ratings > 0)), \
                    f'''Train/Validation/Test set was initialized, use the function load_fold or 
                    set the fold_number parameter on this function'''
            train_set = self.train
            validation_set = self.validation
            test_set = self.test
            fold_loaded = self.fold_loaded
        else:
            assert ((fold_number < self.k_folds) & (len(self.folds_set) > 0)),\
                f'''The maximum number of folds of this dataset is {self.k_folds-1} which 
                    is smaller than {fold_number}'''
            train_set = self.folds_set[fold_number][0]
            validation_set = self.folds_set[fold_number][1]
            test_set = self.folds_set[fold_number][2]
            fold_loaded = fold_number

        trn_s = train_set.num_ratings
        val_s = validation_set.num_ratings
        tst_s = test_set.num_ratings
        total = trn_s + val_s + tst_s
        full_df = self.load_original()
        full = full_df.shape[0]

        print(f'''#### Fold {fold_loaded} statistics ####''')
        print('''--- Training Raw Stats ---''')
        print(f'''Number of users:\t {train_set.num_users}''')
        print(f'''Number of items:\t {train_set.num_items}''')
        print(f'''Number of ratings:\t {trn_s}''')
        print(f'''Average of rating:\t {train_set.global_mean}\n''')

        print('''--- Validation Raw Stats ---''')
        print(f'''Number of users:\t {validation_set.num_users}''')
        print(f'''Number of items:\t {validation_set.num_items}''')
        print(f'''Number of ratings:\t {val_s}''')
        print(f'''Average of rating:\t {validation_set.global_mean}\n''')

        print('''--- Test Raw Stats ---''')
        print(f'''Number of users:\t {test_set.num_users}''')
        print(f'''Number of items:\t {test_set.num_items}''')
        print(f'''Number of ratings:\t {tst_s}''')
        print(f'''Average of rating:\t {test_set.global_mean}\n''')

        print('''--- Dataset Percentage Stats ---''')
        print(f'''Training percentage:\t {trn_s/total}''')
        print(f'''Validation percentage:\t {val_s/total}''')
        print(f'''Test percentage:\t\t {tst_s/total}''')
        print(f'''Full dataset used:\t\t {total/full}\n''')

        print('''--- Dataset Raw Stats ---''')
        print(f'''Total fold size:\t {total}''')
        print(f'''Full Dataset size:\t {full}\n''')
