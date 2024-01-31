# ActStabML

Repository with multi-input graph neural network for modeling metal complexes stability constant

## How to setup enviroment
1. Install anaconda/miniconda package manager https://docs.conda.io/projects/miniconda/en/latest/
2. Create new environment 
```conda env create -n actstabml```
3. Activate this environment
```conda activate actstabml```
4. Now we will install necessary packages </br>
### If you have GPU in your machine
```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
conda install dgl -c dglteam/label/cu118
conda install pytorch-lightning rdkit -c conda-forge
conda install pip tqdm loguru optuna pandas 
pip install notebook numba scikit-learn selfies tensorboard dgllife
```
### If you use non-GPU machine
```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch
conda install pyg -c pyg
conda install dgl -c dglteam 
conda install pytorch-lightning rdkit -c conda-forge
conda install pip tqdm loguru optuna pandas
pip install notebook numba scikit-learn selfies tensorboard dgllife
````
5. To train model on default data
```
5. python3 train.py
```
6. To use model for stability constant prediction for benzene
```
python3 predict.py
```
To change training and prediction data - modify corresponding scripts