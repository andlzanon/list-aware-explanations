import math

import numpy as np
import pandas as pd
from recommenders.evaluation.python_evaluation import merge_ranking_true_pred
from scipy.cluster.hierarchy import cophenet, inconsistent
from scipy.spatial.distance import squareform
from scipy.stats import entropy
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

#### EXPLANATION METRICS #####
def lir_metric(beta: float, user: str, items: list, train_set: pd.DataFrame, col_user: str):
    """
    Linking Interaction Recency (LIR): metric proposed in https://dl.acm.org/doi/abs/10.1145/3477495.3532041
    :param beta: parameter for the exponential decay
    :param user: user id
    :param items: list of items used for each recommendation explanation path
    :param train_set: training set of user item interactions
    :param col_user: user column name
    :return: the LIR metric for the user, the lir for every recommendation is the mean of the lir of every recommendation
        and the lir for every recommendation is the mean of the lir for every item in the explanation path
    """
    train_set_copy = train_set.set_index(col_user)

    interacted = train_set_copy.loc[user]
    interacted = interacted.reset_index()
    last_col = interacted.columns[-1]
    interacted = interacted.sort_values(last_col, ascending=True)
    interacted["lir"] = -1

    # for every item calculate the exponential decay
    min = interacted[last_col].min()
    last_value = min
    last_lir = min
    for i, row in interacted.iterrows():
        # if it is min, then lir is the value
        if row[last_col] == min:
            interacted.at[i, "lir"] = min
        # else if the count is the same repeat the sep, otherwise, calculate new sep
        else:
            if row[last_col] == last_value:
                interacted.at[i, "lir"] = last_lir
            else:
                lir = (1 - beta) * float(last_lir) + beta * float(row[last_col])
                interacted.at[i, "lir"] = lir
                last_value = row[last_col]
                last_lir = lir

    scaler = MinMaxScaler()
    interacted['normalized'] = scaler.fit_transform(
        np.asarray(interacted[interacted.columns[-1]]).astype(np.float64).reshape(-1, 1)).reshape(-1)

    # initialize mean variables
    return interacted[interacted["movieId"].isin([str(i) for i in items])]['normalized'].mean()

def sep_metric(beta: float, props: list, prop_set: pd.DataFrame, memo_sep: dict):
    """
    Shared Entity Popularity (SEP) metric proposed in https://dl.acm.org/doi/abs/10.1145/3477495.3532041
    :param beta: parameter for the exponential decay
    :param props: list of properties used for each recommendation explanation path
    :param prop_set: property set extracted from Wikidata
    :param memo_sep: memoization for sep values across users
    :return: the sep metric for the user, the sep for every recommendation is the mean of the sep of every recommendation
        and the sep for every recommendation is the mean of the sep for every item in the explanation path
    """

    # user variables for the mean sep of each explanation and scaler
    total_sum = 0
    total_n = 0
    scaler = MinMaxScaler()
    # for every list of properties in the user list of explanations
    for expl_props in props:
        # explanation variables for the mean sep of each explanation
        items_sum = 0
        items_n = 0
        # for every property list of each explanation
        for p in expl_props:
            # obtain the most popular link to of the property e.g. link actor from property Brad Pitt
            links = list(set(prop_set[prop_set["obj"] == p]['prop'].values))
            l_memo = list(set(memo_sep.keys()).intersection(set(links)))
            if len(l_memo) > 0:
                memo_df = memo_sep[l_memo[0]]
                p_sep_value = memo_df.loc[p][-1]
            else:
                link_df = prop_set[prop_set["prop"].isin(links)]
                # generate dataset with property as index and count as column
                count_link = pd.DataFrame(link_df.groupby("obj").count())
                count_link = count_link.sort_values(by=count_link.columns[0], ascending=True)
                count_link = pd.DataFrame(count_link[count_link.columns[0]])
                # initialize sep column with value -1
                count_link["sep"] = -1

                # obtain min value so we do not need to calculate every time
                # and initialize the last value and last sep as min according to the base case
                min = count_link[count_link.columns[0]].min()
                last_value = min
                last_sep = min
                for i, row in count_link.iterrows():
                    # if it is min, then lir is the value
                    if row[0] == min:
                        count_link.at[i, "sep"] = min
                    # else if the count is the same repeat the sep, otherwise, calculate new sep
                    else:
                        if row[0] == last_value:
                            count_link.at[i, "sep"] = last_sep
                        else:
                            sep = (1 - beta) * float(last_sep) + beta * float(row[0])
                            count_link.at[i, "sep"] = sep
                            last_value = row[0]
                            last_sep = sep

                # generate normalized sep column
                try:
                    count_link['normalized'] = scaler.fit_transform(
                        np.asarray(count_link[count_link.columns[-1]]).astype(np.float64).reshape(-1, 1)).reshape(-1)
                except ValueError:
                    continue

                p_sep_value = count_link.loc[p][-1]
                for l in links:
                    memo_sep[l] = count_link

            # obtain sep value for the property and calculate mean
            items_sum = items_sum + p_sep_value
            items_n = items_n + 1

        # calculate total mean
        try:
            total_sum = total_sum + (items_sum / items_n)
            total_n = total_n + 1
        except ZeroDivisionError:
            total_n = total_n + 1

    return total_sum / total_n


def etd_metric(explanation_types: list, k: int, total_types: int):
    """
    Metric proposed by Ballocu 2022
    :param explanation_types: list of explanation types used in the explanations
    :param k: number of recommendations
    :param total_types: total number of explanation types in the dataset
    :return: the division between the explanation types in the explanations and the minimum between the k and total_types
    """
    return len(set(explanation_types)) / (min(k, total_types))


#### CLUSTERING METRICS #####
def items_per_cluster(cluster_list: list):
    """
    Explanation metrics related to the number of items in each cluster:
    This function returns, the mean of items per cluster, their std deviation, the entropy and
    an array with each position as the len of the cluster
    :param cluster_list: list with cluster number for each element on the position e.g.: [1,2,2,3,4]
    :return: mean and std of the quantity of items per cluster, entropy and the clusters by themselves
    """
    n_clusters = max(cluster_list)
    items_on_cluster = []
    for i in range(0, n_clusters):
        # get items on cluster, then the attributes of the items on the cluster
        i_cluster = [j for j in range(0, len(cluster_list)) if cluster_list[j] == i + 1]
        items_on_cluster.append(i_cluster)

    cluster_len = np.array([len(c) for c in items_on_cluster])
    return cluster_len.mean(), cluster_len.std(), entropy(cluster_len), cluster_len

def hierarchical_clustering_metrics(linkage_matrix: np.array, verbose=False):
    """
    Function that calculates metrics on the clustering algorithm. The current metrics are:
        - [Cophonet](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cophenet.html)
        - [Inconsistency](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.inconsistent.html)
    :param linkage_matrix: linkage matrix generated by the scipy hierarchical clustering algorithms
    :param verbose: True to show the matrix generated by cophonet and inconsistency metrics
    :return: Cophonet and Inconsistency matrix mean and std as an array of tuples
    """
    sqform = squareform(cophenet(linkage_matrix))
    sqform_mean = sqform.mean()
    sqform_std = sqform_mean.std()

    incon = inconsistent(linkage_matrix)
    incon_mean = incon.mean()
    incon_std = incon.std()

    if verbose:
        print("--- Cophonet Matrix ---")
        print(sqform)
        print("--- Inconsistency Matrix ---")
        print(incon)

    return  (sqform_mean, sqform_std), (incon_mean, incon_std)

def clustering_metrics(X: np.array, labels: np.array, verbose=False):
    """
    Function to output metrics with no ground truth of flat clustering. The metrics implemented are:
        - Silhouette: Compares how close a point is to its own cluster vs. other clusters. Range: [-1, 1]
        - Calinski–Harabasz Index: Ratio of between-cluster variance to within-cluster variance (higher is better).
        - Davies–Bouldin Index: Measures similarity between clusters (lower is better).
    :param verbose: True to print values
    :param X: feature as matrix format
    :param labels: labels generated by the clustering algorithm
    :return: the three scores a tuple following the order on the documentation
    """
    s = silhouette_score(X, labels)
    c = calinski_harabasz_score(X, labels)
    d = davies_bouldin_score(X, labels)
    if verbose:
        print(f'''Silhouette: {s}\n''')
        print(f'''Calinski–Harabasz Index: {c}\n''')
        print(f'''Davies–Bouldin Index: {n}\n''')

    return s, c, d


#### RANKING METRICS #####

def fill_ideal_grid_by_manhattan(values, rows=None, cols=None):
    n = len(values)
    values_copy = values.copy()

    # Auto-determine grid size if not specified
    if n > (rows * cols):
        # Try to make the grid as square as possible
        cols = np.ceil(np.sqrt(n))
        rows = np.ceil(n / cols)

    # Initialize empty grid with NaNs
    positions = []

    # Create list of positions with their Manhattan distance from (0, 0)
    for i in range(int(rows)):
        for j in range(int(cols)):
            dist = i + j
            positions.append(((i+1, j+1), dist))

    # Sort positions by Manhattan distance
    positions.sort(key=lambda x: x[1])
    final_posx = [pos[0][0] for pos in positions][:n]
    final_posy = [pos[0][1] for pos in positions][:n]

    values_copy["x_irank"] = final_posx
    values_copy["y_irank"] = final_posy

    return values_copy

def ndcg_2d(predictions: pd.DataFrame, grid_predictions: pd.DataFrame, test_recs: pd.DataFrame,
               k: int, rating_column: str, alg_name: str, col_user='userId', col_item='movieId',
               alpha=1, beta=1, gama=1, rows=3, columns=2, step_x=1, step_y=1,
               verbose=True, save_results=True):
    """
    Implementation of the 2D-NDCG According to the paper:
    Felicioni, Nicolò, et al. "Measuring the ranking quality of recommendations in a two-dimensional carousel setting."
    Italian Information Retrieval Workshop. Vol. 2947. CEUR-WS, 2021.

    This implementation is based on NDCG of the recommenders algorithm of Microsoft.

    :param predictions: dataframe containing scores for all possible (user item) tuples, therefore, it has
        three columns: user, item and predicted rating
    :param grid_predictions: recommendations on a grid where each row is based on an explanation
    :param test_recs: testing set as pandas dataframe
    :param k: size at k of recommendations to evaluate
    :param alpha: alpha weighting param on y position based on the paper it is set to 1
    :param beta: beta weighting param on x position based on the paper it is set to 1
    :param gama: gama weighting param on user position
    :param rows: row size of the screen
    :param columns: column size of the screen
    :param step_x: number of items on the row (horizontal swipe) shown when user swipes
    :param step_y: number of items on the column shown when user swipes (vertical swipe)
    :param rating_column: rating column name
    :param alg_name: algorithms name that generated the grid rerank
    :param col_user: user column name
    :param col_item: item column name
    :param verbose: True to print the results on console, False otherwise
    :param save_results: True to save the results, False otherwise
    :return: 2d-NDCG value of k
    """

    test_recs_grid = test_recs.groupby('userId', group_keys=False).apply(
        lambda group: fill_ideal_grid_by_manhattan(group, rows=rows, cols=columns))

    df_hit, _, _ = merge_ranking_true_pred(
        test_recs, predictions, col_user=col_user,
        col_item=col_item, relevancy_method='top_k',
        col_prediction=rating_column, k=k
    )

    if df_hit.shape[0] == 0:
        return 0.0

    df_hit = df_hit.merge(test_recs_grid, on=[col_user, col_item], how="outer")

    df_dcg = df_hit.merge(predictions, on=[col_user, col_item]).merge(
        test_recs, on=[col_user, col_item], how="outer", suffixes=("_left", None)
    )

    df_dcg = df_dcg.merge(grid_predictions, on=[col_user, col_item], how="outer")

    df_dcg["rel"] = 2 ** df_dcg[rating_column] - 1
    discfun = np.log2
    df_dcg["dcg"] = df_dcg["rel"] / discfun(alpha * df_dcg["y_rank"] +
                                            beta * df_dcg["x_rank"] +
                                            gama * np.maximum(0, np.ceil((df_dcg["y_rank"] - columns) / step_y)) +
                                            gama * np.maximum(0, np.ceil((df_dcg["x_rank"] - rows) / step_x))
                                            )

    df_idcg = df_dcg.sort_values([col_user, rating_column], ascending=False)
    df_idcg["irank"] = df_idcg.groupby(col_user, as_index=False, sort=False)[
        rating_column
    ].rank("first", ascending=False)
    df_idcg["idcg"] = df_idcg["rel"] / discfun(alpha * df_dcg["y_irank"] +
                                               beta * df_dcg["x_irank"] +
                                               gama * np.maximum(0, np.ceil((df_dcg["y_irank"] - columns) / step_y)) +
                                               gama * np.maximum(0, np.ceil((df_dcg["x_irank"] - rows) / step_x))
                                               )

    df_user = df_dcg.groupby(col_user, as_index=False, sort=False).agg({"dcg": "sum"})

    # Calculate the ideal DCG for each user
    df_user = df_user.merge(
        df_idcg.groupby(col_user, as_index=False, sort=False)
        .head(k)
        .groupby(col_user, as_index=False, sort=False)
        .agg({"idcg": "sum"}),
        on=col_user,
    )

    # DCG over IDCG is the normalized DCG
    df_user["ndcg"] = df_user["dcg"] / df_user["idcg"]
    two_d_ndcg = df_user["ndcg"].mean()
    if verbose: print(f'''{alg_name} - NDCG-2D@{k}: {two_d_ndcg}''')
    return two_d_ndcg