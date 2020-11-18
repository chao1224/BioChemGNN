# BioChem GNN

This repository provides the main Graph Neural Network models for Drug Discovery in `pytorch`.

Currently `BioChem GNN` supports following tasks:

+ Molecule Property Prediction
+ Drug-Target Prediction
+ Drug-Drug Prediction

# Environments

```
conda create -n bcg python=3.7
source activate bcg

conda install -y -c pytorch pytorch=1.6 torchvision
conda install -y -c rdkit rdkit
conda install -y scikit-learn
conda install -y numpy
conda install -y matplotlib
```

To install the current package, now we support the development mode:
Go to home directory of this repo and do `pip install --user -e .`.

# Graph Neural Networks

Now we provide following graph neural networks:

| Model | Paper |
| :---: | :---: |
| Graph Attention Networks (GAT) | [Graph Attention Networks, ICLR 2018](https://arxiv.org/abs/1710.10903) |
| Graph Isomorphism Network (GIN) | [How Powerful are Graph Neural Networks?, ICLR 2019](https://arxiv.org/abs/1810.00826) |
| Directed Message Passing Neural Network (D-MPNN) | [Analyzing Learned Molecular Representations for Property Prediction, ACS JCIM 2019](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237) |

# Motivation

`BioChen GNN` includes sparse and efficient implementations of Graph Neural Network models.
Specifically, the sparse operation is based on the fact that all molecules nodes have very low degree, which allows us to do padding to some degree.

The benefit is that it can avoid the `scatter` operation.
Recall that `scatter` operation is not deterministic and it can bring in some extra issues on some smaller datasets like `delaney`.
About the deterministic-related discussion, feel free to check more details on the `Reproducibility` on pytorch document [here](REPRODUCIBILITY).

And accordingly, the drawback of this repo is that it does not support graph of nodes with higher degrees, like biological knowledge graph.
For those applications, we would recommend another well-organized git repo: [Drug Discovery Platform](https://github.com/DeepGraphLearning/drugdiscovery).

# Acknowledgements

We would like to acknowledge the following related projects.

+ [BioChemGNN_Dense](https://github.com/chao1224/BioChemGNN_Dense):
This is a smart and dense version of `BioChemGNN`, and is only applicable for small-molecule data.
+ [Drug Discovery Platform](https://github.com/DeepGraphLearning/drugdiscovery):
This is a well-organized platform and it also supports large-scale biology knowledge graph.
Yet it includes the `catter` operation, which is non-deterministic.
+ [chemprop](https://github.com/chemprop/chemprop): 
This is the repo for D-MPNN and is the first well-acknowledged repo to use such padding idea for non-determinstic operation.
