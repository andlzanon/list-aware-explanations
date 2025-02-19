import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, name: str, path: str, original_file_name: str,
                 user_column: str, item_column: str, rating_column: str,
                 k_folds=5):
        """
        Dataset class definition. The Dataset class has three main attributes.
        :param name: name of the dataset
        :param path: absolute path of where the dataset is
        :param original_file_name: name of the file with the original full
        :param user_column: name of the user column on the dataset
        :param item_column: name of the item column on the dataset
        :param rating_column: name of the rating column on the dataset
        :param k_folds: number of folds in which the dataset was divided
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
        self.train = pd.DataFrame()
        self.validation = pd.DataFrame()
        self.test = pd.DataFrame()

    def load_original(self) -> pd.DataFrame:
        """
        Load the original dataset without preprocessing, not divided in train/validation/test
        :return: dataframe with the original dataset
        """
        return pd.read_csv(self.__path + '''/''' + self.__original_file_name)

    def load_fold(self, i: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Function to load a fold of the dataset
        :param i: number of the fold to load
        :return: train, validation and test sets as a tuple of DataFrames
        """
        assert(i <= self.k_folds), f'''The dataset {self.__name} has five folds (0 to {self.k_folds-1})'''
        path = self.__path + f'''/folds/{i}/'''
        self.train = pd.read_csv(path + '''train.csv''')
        self.validation = pd.read_csv(path + '''validation.csv''')
        self.test = pd.read_csv(path + '''test.csv''')
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
            assert ((self.train.shape[0] > 0) & (self.validation.shape[0] > 0) & (self.test.shape[0] > 0)), \
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

        trn_s = train_set.shape[0]
        val_s = validation_set.shape[0]
        tst_s = test_set.shape[0]
        total = trn_s + val_s + tst_s
        full_df = self.load_original()
        full = full_df.shape[0]

        print(f'''#### Fold {fold_loaded} statistics ####''')
        print('''--- Training Raw Stats ---''')
        print(f'''Number of users:\t {train_set[self.user_column].unique().size}''')
        print(f'''Number of items:\t {train_set[self.item_column].unique().size}''')
        print(f'''Number of ratings:\t {trn_s}''')
        print(f'''Average of rating:\t {train_set[self.rating_column].mean()}''')
        print(f'''Std of rating:\t\t {train_set[self.rating_column].std()}\n''')

        print('''--- Validation Raw Stats ---''')
        print(f'''Number of users:\t {validation_set[self.user_column].unique().size}''')
        print(f'''Number of items:\t {validation_set[self.item_column].unique().size}''')
        print(f'''Number of ratings:\t {val_s}''')
        print(f'''Average of rating:\t {validation_set[self.rating_column].mean()}''')
        print(f'''Std of rating:\t\t {validation_set[self.rating_column].std()}\n''')

        print('''--- Test Raw Stats ---''')
        print(f'''Number of users:\t {test_set[self.user_column].unique().size}''')
        print(f'''Number of items:\t {test_set[self.item_column].unique().size}''')
        print(f'''Number of ratings:\t {tst_s}''')
        print(f'''Average of rating:\t {test_set[self.rating_column].mean()}''')
        print(f'''Std of rating:\t\t {test_set[self.rating_column].std()}\n''')

        print('''--- Dataset Percentage Stats ---''')
        print(f'''Training percentage:\t {trn_s/total}''')
        print(f'''Validation percentage:\t {val_s/total}''')
        print(f'''Test percentage:\t\t {tst_s/total}''')
        print(f'''Full dataset used:\t\t {total/full}\n''')

        print('''--- Dataset Raw Stats ---''')
        print(f'''Total fold size:\t {total}''')
        print(f'''Full Dataset size:\t {full}\n''')
