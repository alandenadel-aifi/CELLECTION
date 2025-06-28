# CellECTION: An Attention-Based Multiple Instance Learning Approach to Predict Emergent Phenotypes from Single Cell Populations

developed by Hongru Hu: hrhu@ucdavis.edu

![alt text](https://github.com/quon-titative-biology/CELLECTION/blob/main/img/CellECTIVE.png)

Biological systems exhibit emergent phenotypes that arise from the collective behavior of individual components, such as whole-organ functions that arise from the coordinated activity of its individual cells, or organism-level phenotypes that result from the functional interplay of collections of genes in the genome. We present CELLECTION, a deep learning framework that learns to associate subgroups of instances with different emergent phenotypes. We show CELLECTION enables interpretable predictions for heterogeneous tasks, including disease classification, identification of disease-associated cell subtypes, alignment of developmental stages between human model systems, and even predicting relative hand-wing indices across the avian lineage. CELLECTION therefore provides a scalable and flexible framework for identifying key cellular or genetic signatures underlying complex traits in development, disease, and evolution.

---
### Package installation
Please clone this repository:
```command line
git clone https://github.com/quon-titative-biology/CELLECTION
cd CELLECTION
```
PyPI installation will be released with preprint.

---
### Package requirements
scPair is implemented using `torch 2.4.1`, `anndata 0.10.9`, and `scanpy 1.10.3`  under `Python 3.10.15`. 

Users can choose to create the environment provided under this repository [(env file)](https://github.com/quon-titative-biology/CELLECTION/blob/main/environment.yml):
```command line
conda env create --file=environment.yml
```
