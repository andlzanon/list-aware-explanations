# List Aware Explanations

## Description
The main objective of this project is to generate list
aware explanations for recommendations. Consequently, an explanation algorithm 
consider that items are on a list and that explanations may not be only for
single items, but also for a set. 

Furthermore, this explores also the best ways to explain the many
possible sub-lists of recommendations within a set of 
recommendations. Such concept can even be used in carrousels, where tags, or content
are used to explain a row of recommended items.

The intention of generating explanations considering them as lists is for then to create a grid
where each explanation is a row category in the home page os streaming services like Netflix and Spotify, where
recommendations are clustered considering a similar content. The idea is also to adapt and choose an optimal number of 
clusters to optimize user experience. 

## Reproduction
### Environment 
To install the libraries used in this project, use the command: 
    
    pip install requirements

Or create a conda enviroment with the following command:

    conda env create --f requirements.txt

I advise to remove the library [Recommenders](https://github.com/recommenders-team/recommenders) from the requirements 
and installing it without the dependencies, otherwise it will request a different version of 
[Cornac](https://github.com/PreferredAI/cornac) (which was used to implement recommender engines), and previous 
versions of other libraries. Since we are using ```recommenders``` only to evaluate ranking metrics, the dependencies 
are not required and already installed. The version of ```recommenders``` used is 1.2.1.

    pip install --no-deps recommenders

We used [Anaconda](https://www.anaconda.com/) to run the experiments. The version of Python used was the [3.12.3](https://www.python.org/downloads/release/python-3123/).

## Command Line Arguments

Explanations are generated as post-hoc (or agnostic) to the recommender system. As a result, we have to define:

- A recommendation algorithm from Cornac
- A set of explanation algorithms to explain the generated recommendations

These two algorithms should be defined in a JSON file on the experiment configuration folder. An example of experiment 
can be seen in [experiment1.json](datasets/ml-latest-small/experiment_configuration/experiment1.json). 
Each file represents an experiment, and it requires two main components:  a `recommender` and a set of `explainers`.
These two components have a name and a set of parameters. The name a recommender needs to match the Cornac class name for 
a recommendation algorithm and the explanation algorithm class name, that matches the class name from the algorithms
developed in the [explanations](explanations) folder. Similarly, to define the parameters of algorithms they should 
also match the respective implementation parameter names from Cornac and [explanations](explanations) folder, 
respectively. 

After a JSON experiment file is created, it is required to specify the dataset and files used to run the experiment. To 
set them, use the following command line parameters:

- `dataset`: Data set. Either 'ml100k' for the movielens 100k dataset or 'lastfm' for the lastfm dataset


- `split`: Split of the dataset. If 0, then use train/test split set to 80/20, if 1 use K-Fold with train, validation and test sets.


- `start_fold`: Start fold to run the experiments if split is 1


- `end_fold`: End fold to run the experiments if split is 1


- `k_list`: Top K items to generate explanations to evaluate on offline metrics. Separate numbers with space. E.g.: 1 3 5 10


- `n_users`: User to generate only explanations to. Recommendations will be generated to all users


- `rows`: Rows on the grid to generate the NDCG-2D.


- `columns`: Columns on the grid to generate the NDCG-2D


- `rec_model_folder`: Name of the file of the recommendation algorithm model to load


- `experiment_file`: Path for the experiment file configuration

An example of a command-line argument is:

    python main.py --dataset=ml100k --split=0 --k_list=90 --rows=3 --columns=3 --rec_model_folder=BPR --experiment_file=experiment1.json --n_users=3

To run the results for the mobile configuration of the RecSys LBR run:
    
    python main.py --dataset=ml100k --split=0 --k_list=350 --rows=3 --columns=3 --rec_model_folder=BPR --experiment_file=recsys_lbr_mobile1.json --n_users=0
    python main.py --dataset=ml100k --split=0 --k_list=350 --rows=3 --columns=3 --rec_model_folder=BPR --experiment_file=recsys_lbr_mobile2.json --n_users=0
    python main.py --dataset=ml100k --split=0 --k_list=350 --rows=3 --columns=3 --rec_model_folder=BPR --experiment_file=recsys_lbr_mobile3.json --n_users=0

## Project Organization

:file_folder: datasets: file with MovieLens 100k dataset. It has three main folds:
(i)  the cross validation folds, (ii) simple stratified split train/test split and the
(iii) the experiment_configuration folder. Each fold on the cross validation of (i) and
(ii) are subdivided into three other folders:

- :file_folder: explanations: This folder will display the explanations generated by each algorithm
- :file_folder: model: This folder saves cornac models that can be loaded with the ``rec_model_folder`` command line 
argument
- :file_folder: outputs: Contains all recommendations to all users generated by the recommendation algorithm
- :file_folder: results: Contains a JSON with all metrics of the recommendation algorithm and explanation algorithms defined in the 
experiment configuration passed in the command line argument `experiment_file`

Folder (iii) represents the explanation and recommendation algorithms 
that will be used to form the pipeline to generate recommendations associated with explanations. It is a JSON file and
should have two main components: a `recommender` and a set of `explainers`, as described in the 
[Project Organization](#Project-Organization) section. 

:file_folder: knowledge-graphs: files of metadata generated from the Wikidata for items and datasets

:file_folder: dataset_experiment: class of a dataset superclass and specified for each class. This class
is responsible for holding the dataset as a pandas df and cornac Dataset

:file_folder: recommender: implementation of recommender engines on [Cornac](https://github.com/PreferredAI/cornac) library.

:file_folder: notebooks: folder used to learn about the topics applied in this project. This folder is for educational
purposes

:page_facing_up: main.py: main source code to run command line arguments experiments

:page_facing_up: requirements.txt: list of library requirements to run the code
