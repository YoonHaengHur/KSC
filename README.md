# Kernel Synthetic Coupling (KSC)
 
This is the official code repository of the paper [*A Convexified Matching Approach to Imputation and Individualized Inference* (2024)](https://arxiv.org/abs/2407.05372). To implement the KSC method on Python, simply place the `KSC.py` file in the same directory as your code and import the `KSC` class. A tutorial on how to use the KSC class is provided in the Jupyter notebook `tutorial.ipynb`.


### Reproducing the Plots in the Paper
The following Jupyter notebooks reproduce the plots in the paper: 
- `NSW_imputation_linear.ipynb`: reproduces Figure 1 in Section 2
- `NSW_imputation_rbf.ipyb`: reproduces Figure 4 in Section 5
- `NSW_imputation_poly.ipynb`: reproduces Figure 5 in Section 5
- `NSW_CI.ipynb`: reproduces Figures 6 and 7 in Section 5
- `CI_simulation.ipynb`: reproduces Figures 2 and 3 in Section 3.

The first four notebooks analyze the National Supported Work (NSW) demonstration program using the data `nsw_dw.dta` and `psid_controls.dta` in the folder `data/NSW`, which are downloaded from https://users.nber.org/~rdehejia/nswdata2.html. 

### Dependencies
The main module `KSC.py` requires the following Python packages:
- `numpy`
- `POT` (Python Optimal Transport)

The following packages are required to run the Jupyter notebooks:
- `scikit-learn`
- `scipy`
- `pandas`
- `statsmodels`
- `matplotlib`
- `seaborn`

### Contact
If you have any questions or suggestions, please contact [YoonHaeng Hur](https://yoonhaenghur.github.io/) via email yoonhaenghur@uchicago.edu.



