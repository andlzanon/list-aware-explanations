import argparse
import cornac
import numpy as np
import pandas as pd

from dataset_experiment.movielens100k import MovieLens100K
from explanations.explod import ExpLOD


def user_semantic_profile(prop_set: pd.DataFrame, historic: list) -> dict:
    """
    Generate the user semantic profile, where all the values of properties (e.g.: George Lucas, action films, etc)
    are ordered by a score that is calculated as:
        score = (npi/i) * log(N/dft)
    where npi are the number of edges to a value, i the number of interacted items,
    N the total number of items and dft the number of items with the value
    :param prop_set: knowledge graph with item information as triples ('item title', 'prop', 'attribute') with
        the index as the item id
    :param historic: list of the items interacted by a user
    :return: dictionary with properties' values as keys and scores as values
    """

    # create npi, i and n columns
    interacted_props = prop_set.loc[prop_set.index.isin(historic)].copy()
    interacted_props['npi'] = interacted_props.groupby(prop_set.columns[-1])[prop_set.columns[-1]].transform(
        'count')
    interacted_props['i'] = len(historic)
    interacted_props['n'] = len(prop_set.index.unique())

    # get items per property on full dbpedia/wikidata by dropping the duplicates with same item id and prop value
    # therefore, a value that repeats in the same item is ignored
    items_per_obj = prop_set.reset_index().drop_duplicates(
        subset=[prop_set.columns[0], prop_set.columns[-1]]).set_index(
        prop_set.columns[-1])
    df_dict = items_per_obj.index.value_counts().to_dict()

    # generate the dft column based on items per property and score column base on all new created columns
    interacted_props['dft'] = interacted_props.apply(lambda x: df_dict[x[prop_set.columns[-1]]], axis=1)

    interacted_props['score'] = (interacted_props['npi'] / interacted_props['i']) * (
        np.log(interacted_props['n'] / interacted_props['dft']))

    # generate the dict
    interacted_props.reset_index(inplace=True)
    interacted_props = interacted_props.set_index(prop_set.columns[-1])
    fav_prop = interacted_props['score'].to_dict()

    return fav_prop

def __explod_ranked_paths(prop_set: pd.DataFrame, ranked_items: list, items_historic: list, semantic_profile: dict, user: int,
                          train_set: pd. DataFrame):
    # get properties from historic and recommended items
    hist_props = prop_set.loc[items_historic]
    prop_cols = prop_set.columns
    for r in ranked_items:
        rec_props = prop_set.loc[int(r)]

        # check properties on both sets
        intersection = pd.Series(list(set(hist_props['obj']).intersection(set(rec_props['obj']))))

        # get properties with max value
        max = -1
        max_props = []
        for pi in intersection:
            value = semantic_profile[pi]
            if value > max:
                max = value
                max_props.clear()
                max_props.append(pi)
            elif value == max:
                max_props.append(pi)

        # build sentence
        train_c = train_set.copy()
        train_c = train_c.set_index(train_set.columns[0])

        user_df = train_c.loc[str(user)]
        user_item = user_df[
            user_df[user_df.columns[0]].isin(list(hist_props[hist_props['obj'].isin(max_props)].index.unique().astype(str)))]
        hist_ids = list(user_item.sort_values(by=user_item.columns[-1], ascending=False)[:3][user_item.columns[0]].astype(int))
        hist_names = hist_props.loc[hist_ids][prop_cols[0]].unique()
        try:
            rec_name = prop_set.loc[int(r)]['title'].unique()[0]
        except AttributeError:
            rec_name = prop_set.loc[int(r)][prop_cols[0]]

        print("\nRecommended Item: " + str(r) + ": " + str(rec_name))
        origin = "Because you watched "
        # check for others with same value
        for i in hist_names:
            origin = origin + "\"" + i + "\"; "
        origin = origin[:-2]

        path_sentence = " that share the attribute "
        for n in max_props:
            path_sentence = path_sentence + "\"" + n + "\" "
        destination = ", watch \"" + rec_name + "\" that has the same attribute"
        print(origin + path_sentence + destination)

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

ml = MovieLens100K(gen_dataset=True)
ds = ml.load_fold(-1)
ml.fold_percentage()

bpr = cornac.models.BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123, verbose=True)
bpr.fit(train_set=ml.train, val_set=ml.validation)

user_id = '5'
recs = bpr.recommend(user_id=user_id, k=10, train_set=ml.train, remove_seen=True)

u_hist = [next((int(k) for k, v in ml.train.iid_map.items() if v == u_item), None) for u_item in ml.train.chrono_user_data[ml.train.uid_map[user_id]][0]]
sem_pro = user_semantic_profile(ml.prop_set, u_hist)
__explod_ranked_paths(ml.prop_set, list(recs), u_hist, sem_pro, int(user_id), ml.load_fold_asdf(-1)[0])

print("### class ###")
explod = ExpLOD(ml, bpr)
explod.user_explanation(user='5', top_k=10, remove_seen=True)


if args.dataset == "ml":
    ml = MovieLens100K()
    ds = ml.load_fold(args.fold)
