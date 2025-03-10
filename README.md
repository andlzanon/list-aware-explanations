# List Aware Explanations

## Description
The main objective of this project is to generate list
aware explanations for recommendations. Consequently, this explanation algorithm 
consider that items are on a list and that explanations may not be only for
single items, but also for a set. 

Furthermore, this explores also the best ways to explain the many
possible sub-lists of recommendations within a set of 
recommendations. Such concept can even be used in carrousels, where tags
are used to explain a row of recommended items.

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

## Project Organization

:file_folder: datasets: file with MovieLens 100k, 
folds of cross validation and simple stratified split.
In addition, there are experiments outputs and results.

:file_folder: knowledge-graphs: files of metadata generated from the Wikidata for items and datasets

:file_folder: dataset_experiment: class of a dataset superclass and specified for each class. This class
is responsible for holding the dataset as a pandas df and cornac Dataset

:file_folder: recommender: implementation of recommender engines on [Cornac](https://github.com/PreferredAI/cornac) library.

:file_folder: notebooks: folder used to learn about the topics applied in this project. This folder is for educational
purposes

:page_facing_up: main.py: main source code to run command line arguments experiments

:page_facing_up: requirements.txt: list of library requirements to run the code