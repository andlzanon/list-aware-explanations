from datasets.dataset import Dataset

class MovieLens100K(Dataset):

    __K_FOLDS = 5

    def __init__(self):
        super().__init__(name="MovieLens100k", path="./datasets/ml-latest-small",
                         user_column="userId", item_column="movieId", rating_column="rating",
                         original_file_name="ratings_processed.csv", k_folds=self.__K_FOLDS)
