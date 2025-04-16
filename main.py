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

# dataset definition
ml = MovieLens100K(gen_dataset=True)
ds = ml.load_fold(-1)
ml.fold_percentage()

user_id = '5'

# declaration of the recommender systems
rec = RecommenderSystem(model=cornac.models.BPR(k=200, max_iter=200, learning_rate=0.01,
                                                lambda_reg=1e-3, seed=42, verbose=True),
                        dataset=ml, remove_seen=True,
                        load_path="BPR"
                        #load_path=None
                        )
rec.fit_model(save=True)

# declaration of the explanation algorithms
cluter_rel = HierarchicalClustering(ml, rec.model, n_clusters=5, method='average', top_n=2, hitems_per_attr=2,
                                            metric="cosine", criterion="maxclust", vec_method='relevance')

cluter_cosine = HierarchicalClustering(ml, rec.model, n_clusters=5, method='average', top_n=2, hitems_per_attr=2,
                                               metric="cosine", criterion="maxclust", vec_method='binary')

cluter_euclid = HierarchicalClustering(ml, rec.model, n_clusters=5, method='ward', top_n=2, hitems_per_attr=2,
                                               metric="euclidean", criterion="maxclust", vec_method='binary')

explod = ExpLOD(ml, rec.model)

# definition of parameters
users = ml.get_users()[:10]
users = ['5']
explain = True
res = {}
k_list = [10]

for user_id in users:
    if explain:
        for k in k_list:
            print(f'''#### User {user_id} Explanations #### \n''')
            rec_list = rec.recommend_to_user(user_id=user_id, k=k)

            print("--- Clustering Algorithm with TF-IDF ---\n")
            cluter_rel_expls = cluter_rel.user_explanation(user=user_id, top_k=k, remove_seen=True,
                                                           verbose=True, show_dendrogram=False)
            res[cluter_rel.model_name] = cluter_rel_expls

            print("--- Clustering Algorithm with Cosine Similarity ---\n")
            cluter_cosine_expls = cluter_cosine.user_explanation(user=user_id, top_k=k, remove_seen=True,
                                                                 verbose=True, show_dendrogram=False)
            res[cluter_cosine.model_name] = cluter_cosine_expls

            print("--- Clustering Algorithm with Euclidean Similarity ---\n")
            cluter_euclid_expls = cluter_euclid.user_explanation(user=user_id, top_k=k, remove_seen=True,
                                                                 verbose=True, show_dendrogram=False)
            res[cluter_euclid.model_name] = cluter_euclid_expls

            print("--- ExpLOD Algorithm ---")
            explod_expls = explod.user_explanation(user=user_id, top_k=k, remove_seen=True,
                                                   verbose=True, top_n=2, hitems_per_attr=2)
            res[explod.model_name] = explod_expls

            print()

# printing offline and explanation metrics
all_metrics = rec.run_experiment(k_list, res, rows=3, cols=2, verbose=False, save_results=True)
for key, value in all_metrics.items():
    print(f'''{key}: {value}''')

args = parser.parse_args()

if args.dataset == "ml":
    ml = MovieLens100K()
    ds = ml.load_fold(args.fold)
