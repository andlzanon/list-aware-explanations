import cornac
from explanations.explanation import ExplanationAlgorithm
from explanations.explod import ExpLOD
from explanations.explod_rows import ExpLODRows
from explanations.hierarchical_clustering import HierarchicalClustering
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
    if explainer_name == "HierarchicalClustering":
        return HierarchicalClustering(ds_expr, rec_alg, expr_file, top_k, n_clusters=explainer_params["n_clusters"],
                                      method=explainer_params["method"], top_n=explainer_params["top_n"],
                                      hitems_per_attr=explainer_params["hitems_per_attr"],
                                      metric=explainer_params["metric"], criterion=explainer_params["criterion"],
                                      vec_method=explainer_params["vec_method"], n_users=n_users)
    elif explainer_name == "ExpLOD":
        return ExpLOD(ds_expr, rec_alg, expr_file, top_k, top_n=explainer_params["top_n"],
                      hitems_per_attr=explainer_params["hitems_per_attr"], n_users=n_users)
    elif explainer_name == "ExpLODRows":
        try:
            n_clusters = explainer_params["n_clusters"]
        except KeyError:
            n_clusters = None

        return ExpLODRows(ds_expr, rec_alg, expr_file, top_k, top_n=explainer_params["top_n"],
                        n_clusters=n_clusters, hitems_per_attr=explainer_params["hitems_per_attr"],
                        n_users=n_users)
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