# OMLA: An Oracle-less Machine Learning-based Attack on Logic Locking
## Contact Info
This repository is the official implementation of the experiments in the following paper:

L. Alrahis, S. Patnaik, M. Shafique and O. Sinanoglu, "OMLA: An Oracle-less Machine Learning-based Attack on Logic Locking," in *IEEE TCAS II: Express Briefs*, doi: 10.1109/TCSII.2021.3113035.

[IEEE Link](https://ieeexplore.ieee.org/document/9539868) 

**Contact**

Lilas Alrahis (lma387@nyu.edu)
## Citation
If you make use of the code/experiment or OMLA algorithm in your work, please cite our paper (Bibtex below).
```
@ARTICLE{9539868,
  author={Alrahis, Lilas and Patnaik, Satwik and Shafique, Muhammad and Sinanoglu, Ozgur},
  journal={IEEE Transactions on Circuits and Systems II: Express Briefs}, 
  title={{OMLA}: An Oracle-less Machine Learning-based Attack on Logic Locking}, 
  year={2021},
  pages={1-1},
  doi={10.1109/TCSII.2021.3113035}}
```
### Overview 
OMLA is an oracle-less attack on traditional logic locking which maps the problem of resolving the key-bit value to subgraph classification. OMLA extracts a small subgraph for each key-gate from the locked netlist. The enclosing subgraphs capture the characteristics associated with the key-bit values of the key-gates. Therefore, the label of a subgraph can also be considered the key-bit value.

![OMLA Concept](./OMLA.png)

