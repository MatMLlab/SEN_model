# model_test_for_nmi
This is a repository for NMI reviewers.

## Description
This a deep learning model for structure symmetry description, material property prediction, and material design, which is based on the symmetry-enhanced capsule transformer.

## Development Environment
### Package dependencies

- pymatgen==2022.4.19
- matplotlib>=3.0.3
- pandas>=1.0.5
- numpy>=1.16.2
- scikit_learn>=0.20.4
- keras==2.8.0
- sonnet==1.35
- tensorflow==2.8.0
- tensorflow_probability==0.8.0
- tensorflow_datasets==1.3.0

### Package Installation
We can install all neccessary packages according to 'pip' or 'conda'. 
All data can be downloaded from Material Project (https://materialsproject.org/). 
All data involved in this work will be available upon request.
If you need test data, please contact us and provide the transmission address or email that can receive large files, and we will send you complete data. 

## About the Code
- The training and testing process are defined in `main.py`.
- Related data is in `data file`, and we also prepared the code `data/data.py` to get the dataset from `Material Project`.
- The capsule transformer is defined in `cap_block.py`.
- Related models are defined in `models file`.
- The visualization process in the manuscript is in `plot file`.

## Thanks for your time and attention.
