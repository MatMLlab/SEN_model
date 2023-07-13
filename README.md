[10.5281/zenodo.8142678](https://zenodo.org/badge/DOI/10.5281/zenodo.8142678.svg)
# SEN Model
This is a repository for SEN model.
Detailed description is prepared in `Cap_Models` file.

## Description
This is a deep learning model for structure symmetry description, material property prediction, and material design, which is based on the symmetry-enhanced capsule transformer.

## Development Environment
### Package dependencies

- Please make sure about your running enviornment and package version.
- Linux (ubantu)>=18.04
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
We can install all neccessary packages according to 'pip' or 'conda' with short time. 
All data can be downloaded from Material Project (https://materialsproject.org/). 
All data involved in this work will be available upon request.
If you need full data, you can download from Material Project website via `data/data.py`.
Or please contact us and provide the transmission address or email that can receive large files, and we will send you complete data. 

## About the Code
- The training and testing process are defined in `main.py`.
- The feature extraction block is defined in `cap_block.py`.
- The symmetry perception block is defined in `models/che_env_block.py`.
- Related data is in `data file`, and we also prepared the code `data/data.py` to get the dataset from `Material Project`.
- The capsule transformer is defined in `cap_block.py`.
- Related models are defined in `models file`.
- The visualization process and result plots in the manuscript are in `plot file`.
- The running time of one epoch with one batch is less than 1s under NVIDIA GV-100 GPU.

## License
This project is covered under the Apache 2.0 License.

## Thanks for your time and attention.

## References
- Atz, K., Grisoni, F. & Schneider, G. Geometric deep learning on molecular representations. Nat. Mach. Intell. 3, 1023–1032 (2021).
- Kosiorek, A. R., Sabour, S., Teh, Y. W. & Hinton, G. E. Stacked capsule autoencoders. Preprint at http://arXiv.org/abs/1906.06818 (2019). 
- Xie, T. & Grossman, J. C. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. Phys. Rev. Lett. 120, 145301 (2018). 
- Chen, C., Ye, W., Zuo, Y., Zheng, C. & Ong, S. P. Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals. Chem. Mater. 31, 3564–3572 (2019). 
- Liang, C.,  Jiang, H.,  Lin, S.,  Li, H., &  Wang, B. Intelligent generation of evolutionary series in a time‐variant physical system via series pattern recognition. Adv. Intell. Sys.(2020)
