# RecSys LBR

This paper propose to cluster items based on their metadata to automatically generate carousels and corresponding 
descriptive titles. Each carousel represents a cluster of items sharing similar metadata, which is then used to create 
the carousel’s descriptive title.

## 🔖 Citation

If you use any of the results or code from these experiments please cite us:

Paper submitted to the LBR of RecSys '25 in Prague. 

## 💻 RecSys LBR Experiments Commands 

Here are the command line arguments used that resulted in the explanations and results reported on the paper.

To see the meaning of each parameter and how to build your own, please check section "Command Line Arguments" in the 
[README.md](README.md) of the project.

Each set of commands represent: The first to run the **baseline model**, the second one changing the **encodings** 
and the third changing the **clustering algorithm**.

To run the experiments for the mobile configuration of the RecSys LBR for the MovieLens dataset run:
    
    python main.py --dataset=ml100k --split=0 --k=350 --rows=3 --columns=3 --rec_model_folder=BPR --experiment_file=recsys_lbr_mobile1.json --n_users=0
    python main.py --dataset=ml100k --split=0 --k=350 --rows=3 --columns=3 --rec_model_folder=BPR --experiment_file=recsys_lbr_mobile2.json --n_users=0
    python main.py --dataset=ml100k --split=0 --k=350 --rows=3 --columns=3 --rec_model_folder=BPR --experiment_file=recsys_lbr_mobile3.json --n_users=0

To run the experiments for the PC configuration of the RecSys LBR for the MovieLens dataset run:
    
    python main.py --dataset=ml100k --split=0 --k=500 --rows=3 --columns=6 --rec_model_folder=BPR --experiment_file=recsys_lbr_pc1.json --n_users=0
    python main.py --dataset=ml100k --split=0 --k=500 --rows=3 --columns=6 --rec_model_folder=BPR --experiment_file=recsys_lbr_pc2.json --n_users=0
    python main.py --dataset=ml100k --split=0 --k=500 --rows=3 --columns=6 --rec_model_folder=BPR --experiment_file=recsys_lbr_pc3.json --n_users=0

To run the experiments for the mobile configuration of the RecSys LBR run for the LastFM dataset run:
    
    python main.py --dataset=lastfm --split=0 --k=300 --rows=2 --columns=2 --rec_model_folder=BPR --experiment_file=recsys_lbr_mobile1.json --n_users=0
    python main.py --dataset=lastfm --split=0 --k=300 --rows=2 --columns=2 --rec_model_folder=BPR --experiment_file=recsys_lbr_mobile2.json --n_users=0
    python main.py --dataset=lastfm --split=0 --k=300 --rows=2 --columns=2 --rec_model_folder=BPR --experiment_file=recsys_lbr_mobile3.json --n_users=0

To run the experiments for the PC configuration of the RecSys LBR run for the LastFM dataset run:
    
    python main.py --dataset=lastfm --split=0 --k=250 --rows=2 --columns=5 --rec_model_folder=BPR --experiment_file=recsys_lbr_pc1.json --n_users=0
    python main.py --dataset=lastfm --split=0 --k=250 --rows=2 --columns=5 --rec_model_folder=BPR --experiment_file=recsys_lbr_pc2.json --n_users=0
    python main.py --dataset=lastfm --split=0 --k=250 --rows=2 --columns=5 --rec_model_folder=BPR --experiment_file=recsys_lbr_pc3.json --n_users=0


## 📊 Results 

### RecSys LBR Results

As in Section [RecSys LBR Experiments Commands](#RecSys-LBR-Experiments-Commands)
we executed 3 commands for each dataset to obtain explanations and result metrics reported. Here are the shortcut to 
the generated carousels with descriptive title explanations and resulted metric for each command of the previous section. 

#### MovieLens
- Explanations
  - Mobile: 
    - [Baseline](datasets/ml-latest-small/stratified_split/explanations/recsys_lbr_mobile1)
    - [Encodings](datasets/ml-latest-small/stratified_split/explanations/recsys_lbr_mobile2)
    - [Clustering Algorithms](datasets/ml-latest-small/stratified_split/explanations/recsys_lbr_mobile3)
  - PC
    - [Baseline](datasets/ml-latest-small/stratified_split/explanations/recsys_lbr_pc1)
    - [Encodings](datasets/ml-latest-small/stratified_split/explanations/recsys_lbr_pc2)
    - [Clustering Algorithms](datasets/ml-latest-small/stratified_split/explanations/recsys_lbr_pc3)
- Results
  - Mobile: 
    - [Baseline](datasets/ml-latest-small/stratified_split/results/recsys_lbr_mobile1)
    - [Encodings](datasets/ml-latest-small/stratified_split/results/recsys_lbr_mobile2)
    - [Clustering Algorithms](datasets/ml-latest-small/stratified_split/results/recsys_lbr_mobile3)
  - PC
    - [Baseline](datasets/ml-latest-small/stratified_split/results/recsys_lbr_pc1)
    - [Encodings](datasets/ml-latest-small/stratified_split/results/recsys_lbr_pc2)
    - [Clustering Algorithms](datasets/ml-latest-small/stratified_split/results/recsys_lbr_pc3)

#### LastFM
- Explanations
  - Mobile: 
    - [Baseline](datasets/hetrec2011-lastfm-2k/stratified_split/explanations/recsys_lbr_mobile1)
    - [Encodings](datasets/hetrec2011-lastfm-2k/stratified_split/explanations/recsys_lbr_mobile2)
    - [Clustering Algorithms](datasets/hetrec2011-lastfm-2k/stratified_split/explanations/recsys_lbr_mobile3)
  - PC
    - [Baseline](datasets/hetrec2011-lastfm-2k/stratified_split/explanations/recsys_lbr_pc1)
    - [Encodings](datasets/hetrec2011-lastfm-2k/stratified_split/explanations/recsys_lbr_pc2)
    - [Clustering Algorithms](datasets/hetrec2011-lastfm-2k/stratified_split/explanations/recsys_lbr_pc3)
- Results
  - Mobile: 
    - [Baseline](datasets/hetrec2011-lastfm-2k/stratified_split/results/recsys_lbr_mobile1)
    - [Encodings](datasets/hetrec2011-lastfm-2k/stratified_split/results/recsys_lbr_mobile2)
    - [Clustering Algorithms](datasets/hetrec2011-lastfm-2k/stratified_split/results/recsys_lbr_mobile3)
  - PC
    - [Baseline](datasets/hetrec2011-lastfm-2k/stratified_split/results/recsys_lbr_pc1)
    - [Encodings](datasets/hetrec2011-lastfm-2k/stratified_split/results/recsys_lbr_pc2)
    - [Clustering Algorithms](datasets/hetrec2011-lastfm-2k/stratified_split/results/recsys_lbr_pc3)
