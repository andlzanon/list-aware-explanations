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

ml = MovieLens100K()
ds = ml.load_fold(args.fold)
ml.load_all_folds()
ml.fold_percentage(0)
ml.fold_percentage(1)
ml.fold_percentage(2)
ml.fold_percentage(3)
ml.fold_percentage(4)

print('--- Train Set ---')
print('Number of users: {}'.format(ml.train.num_users))
print('Number of items: {}'.format(ml.train.num_items))
print('Number of ratings: {}'.format(ml.train.num_ratings))

print(ml.test.global_mean)
print(ml.test.max_rating)
print(ml.test.min_rating)
print(ml.test.dok_matrix)

if args.dataset == "ml":
    ml = MovieLens100K()
    ds = ml.load_fold(args.fold)
