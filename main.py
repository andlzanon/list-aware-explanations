import argparse
import cornac
from cornac.data import Dataset
from datasets.movielens100k import MovieLens100K

parser = argparse.ArgumentParser()

# required arguments
parser.add_argument("--mode",
                    type=str,
                    default="recommend",
                    help="Set 'recommend' to run recommendation algorithms or 'explain' to "
                         "run explanation algorithms")

parser.add_argument("--dataset",
                    type=str,
                    default="ml100k",
                    help="Data set. Either 'ml' for the movielens dataset or 'lastfm' for the lastfm dataset")

parser.add_argument("--fold",
                    type=int,
                    default=4,
                    help="Fold to start the experiment")

parser.add_argument("--alg",
                    type=str,
                    default="None",
                    help="Algorithm to run")

args = parser.parse_args()

ds = None
train_set = None
validation_set = None
test_set = None
seed = 42

ml = MovieLens100K()
ds = ml.load_fold(args.fold)
train_set = Dataset.from_uirt(ml.get_cornac_train(), seed=seed)
validation_set = Dataset.from_uirt(ml.get_cornac_validation(), seed=seed)
test_set = Dataset.from_uirt(ml.get_cornac_test(), seed=seed)
print(ml.fold_percentage())

if args.dataset == "ml":
    ml = MovieLens100K()
    ds = ml.load_fold(args.fold)
    train_set = Dataset.from_uirt(ml.get_cornac_train(), seed=seed)
    validation_set = Dataset.from_uirt(ml.get_cornac_validation(), seed=seed)
    test_set = Dataset.from_uirt(ml.get_cornac_test(), seed=seed)

    print('--- Train Set ---')
    print('Number of users: {}'.format(train_set.num_users))
    print('Number of items: {}'.format(train_set.num_items))
    print('Number of ratings: {}'.format(train_set.num_ratings))
