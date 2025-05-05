import argparse
import json
from utils import create_explainer, create_recommender
from dataset_experiment.movielens100k import MovieLens100K
from recommender.recommender_system import RecommenderSystem

parser = argparse.ArgumentParser()

parser.add_argument("--dataset",
                    type=str,
                    default="ml100k",
                    help="Data set. Either 'ml100k' for the movielens 100k dataset or 'lastfm' for the lastfm dataset")

parser.add_argument("--split",
                    type=int,
                    default=0,
                    help="Split of the dataset. If 0, then use train/test split set to 80/20, if 1 use K-Fold with "
                         "train, validation and test sets.")

parser.add_argument("--start_fold",
                    type=int,
                    default=0,
                    help="Start fold to run the experiments if split is 1")

parser.add_argument("--end_fold",
                    type=int,
                    default=0,
                    help="End fold to run the experiments if split is 1")

parser.add_argument("--k_list",
                    type=str,
                    default="10",
                    help="Top K items to generate explanations to evaluate on offline metrics. Separate numbers"
                         "with space. E.g.: 1 3 5 10")

parser.add_argument("--n_users",
                    type=int,
                    default=0,
                    help="User to generate only explanations to. Recommendations will be generated to all users")

parser.add_argument("--rows",
                    type=int,
                    default=3,
                    help="Rows on the grid to generate the NDCG-2D.")

parser.add_argument("--columns",
                    type=int,
                    default=2,
                    help="Columns on the grid to generate the NDCG-2D")

parser.add_argument("--rec_model_folder",
                    type=str,
                    default="None",
                    help="Name of the file of the recommendation algorithm model to load")

parser.add_argument("--experiment_file",
                    type=str,
                    default="None",
                    help="Path for the experiment file configuration")

args = parser.parse_args()

# get start and end fold, when -1 it uses simple train/test split
ds = None
if not args.split:
    sf = -1
    ed = 0
else:
    sf = int(args.start_fold)
    ed = int(args.end_fold) + 1

# create dataset
ds_folder = ""
if args.dataset == "ml100k":
    ds = MovieLens100K(gen_dataset=True)
    ds_folder = "ml-latest-small"

# load file of experiment configuration
file_path =  f'''./datasets/{ds_folder}/experiment_configuration/''' + args.experiment_file
with open(file_path, 'r') as file:
    expr = json.load(file)

# get the recommender system name
rec_name = expr["recommender"]["name"]
rec_params = expr["recommender"]["parameters"]

# if the recommender has not been saved (equal to "None"), set to None
if args.rec_model_folder == "None":
    args.rec_model_folder = None

# get the k list from command line argument
k_list = [int(x) for x in args.k_list.split(" ")]
n_users = args.n_users

# for every fold
for fold in range(sf, ed):
    # create result file as result json dict and load fold
    res = {}
    ds.load_fold(fold)

    # create the cornac recommendation model, recommender system and fit the model
    rec = RecommenderSystem(model=create_recommender(rec_name, rec_params),
                            dataset=ds, remove_seen=True, load_path=args.rec_model_folder)
    rec.fit_model(save=True)

    # for every k create an explainer based on the experiment configuration file, generate explanation and get results
    for k in k_list:
        for expl in expr["explainers"]:
            expl_name = expl["name"]
            expl_params = expl["parameters"]

            explainer = create_explainer(expl_name, expl_params, ds, rec.model, k, n_users)
            expl_alg_results, _ = explainer.all_users_explanations(remove_seen=True, verbose=True)
            res[explainer.model_name] = expl_alg_results

    # run offline experiments
    expr_out = rec.run_experiment(k_list, res, args.experiment_file, n_users=n_users,
                       rows=args.rows, cols=args.columns, verbose=False, save_results=True)
    print(expr_out)
