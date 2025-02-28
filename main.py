import argparse
import cornac

from dataset_experiment.movielens100k import MovieLens100K
from explanations.explod import ExpLOD
from explanations.hierarchical_clustering import HierarchicalClustering
from recommender.recommender_system import RecommenderSystem

parser = argparse.ArgumentParser()

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

ml = MovieLens100K(gen_dataset=True)
ds = ml.load_fold(-1)
ml.fold_percentage()

user_id = '5'

rec = RecommenderSystem(model=cornac.models.BPR(k=200, max_iter=200, learning_rate=0.01,
                                                lambda_reg=1e-3, seed=42, verbose=True),
                        dataset=ml, remove_seen=True,
                        load_path="BPR"
                        #load_path=None
                        )

rec.fit_model(save=True)
rec_list = rec.recommend_to_user(user_id=user_id, k=10)
print(rec_list)

explod = ExpLOD(ml, rec.model)
explod_expls = explod.user_explanation(user=user_id, top_k=10, remove_seen=True,
                                verbose=False, top_n=3, hitems_per_attr=2)

cluter = HierarchicalClustering(ml, rec.model, n_clusters=5, method='ward', criterion="maxclust")
cluter_expls = cluter.user_explanation(user=user_id, top_k=10, remove_seen=True,
                                verbose=False)

for key in explod_expls.keys():
    print("\nRecommended Item: " + str(key))
    print(explod_expls[key])
    print()

rec.run_experiment([10], True)

args = parser.parse_args()

if args.dataset == "ml":
    ml = MovieLens100K()
    ds = ml.load_fold(args.fold)
