import argparse
from datasets.movielens100k import MovieLens100K

ml = MovieLens100K()
ml.load_fold(4)
ml.load_all_folds()

ml.fold_percentage()
ml.fold_percentage(0)
