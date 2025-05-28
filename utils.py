import cornac
from explanations.explanation import ExplanationAlgorithm
from explanations.explod import ExpLOD
from explanations.explod_rows import ExpLODRows
from explanations.clustering import Clustering
from cornac.models import Recommender

from dataset_experiment.dataset_experiment import DatasetExperiment

def create_explainer(explainer_name: str, explainer_params: dict, ds_expr: DatasetExperiment,
                     rec_alg: Recommender, expr_file: str, top_k: int, n_users: int) -> ExplanationAlgorithm:
    """
    Create an Explainer algorithm instance based on the parameters of explainer_params, the name comes from the
    experiment file
    :param explainer_name: name of the explanation algorithm
    :param explainer_params: dict with the explainers parameters
    :param ds_expr: dataset of the explainer
    :param rec_alg: recommendation algorithm
    :param expr_file: name of the experiment file configuration
    :param top_k: top k items to explain
    :param n_users: number of users to generate explanations to. If 0 runs to all users
    :return: the explanation algorithm object instance
    """
    if explainer_name == "Clustering":
        try:
            method = explainer_params["method"]
            criterion = explainer_params["criterion"]
        except KeyError:
            method = None
            criterion = None

        try:
            alpha = explainer_params["alpha"]
            beta = explainer_params["beta"]
        except KeyError:
            alpha = None
            beta = None

        return Clustering(ds_expr, explainer_params["alg"], rec_alg, expr_file, top_k, n_clusters=explainer_params["n_clusters"],
                          method=method, top_n=explainer_params["top_n"],
                          hitems_per_attr=explainer_params["hitems_per_attr"],
                          metric=explainer_params["metric"], criterion=criterion,
                          vec_method=explainer_params["vec_method"], n_users=n_users,
                          alpha=alpha, beta=beta)

    elif explainer_name == "ExpLOD":
        return ExpLOD(ds_expr, rec_alg, expr_file, top_k, top_n=explainer_params["top_n"],
                      hitems_per_attr=explainer_params["hitems_per_attr"], n_users=n_users)
    elif explainer_name == "ExpLODRows":
        try:
            n_clusters = explainer_params["n_clusters"]
        except KeyError:
            n_clusters = None

        try:
            alpha = explainer_params["alpha"]
            beta = explainer_params["beta"]
        except KeyError:
            alpha = 0.5
            beta = 0.5

        try:
            all_props_on_items = explainer_params["all_props_on_items"]
        except KeyError:
            all_props_on_items = False

        return ExpLODRows(ds_expr, rec_alg, expr_file, top_k, top_n=explainer_params["top_n"],
                        n_clusters=n_clusters, hitems_per_attr=explainer_params["hitems_per_attr"],
                        n_users=n_users, alpha=alpha, beta=beta, all_props_on_items=all_props_on_items)
    else:
        raise ValueError("Explainer name of the algorithm is not correct.")

def create_recommender(model_name: str, model_params: dict) -> Recommender:
    """
    Create a Recommender cornac model based on the parameters of model_params, the name comes from the experiment file
    :param model_name: name of the model to instantiate
    :param model_params: dict with the model parameters
    :return: the cornac model
    """
    if model_name == "BPR":
        return cornac.models.BPR(k=model_params["k"], max_iter=model_params["max_iter"],
                                 learning_rate=model_params["learning_rate"],lambda_reg=model_params["lambda_reg"],
                                 seed=model_params["seed"], verbose=model_params["verbose"])
    else:
        raise ValueError("Recommender name of the algorithm is not correct.")