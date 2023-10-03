# Classifying Twin gas sensor arrays Data Set
###### Author: Yiming Bian (yiming.bian1217@gmail.com)
This repo works on the classification of four types of gas: Ethylene, Ethanol, Carbon Monoxide, and Methane. The dataset called "Twin gas sensor arrays Data Set" is generously shared to public by University of California Irvine (UCI). It is available on [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/361/twin+gas+sensor+arrays). The detailed description of this data set can be found there. Again, they take the full credit for collecting and sharing these high-quality data. We appreciate their remarkable contribution! 

The purpose of this repo is exploring the dimensionality reduction potential of the original data while keeping the classification accuracies comparably high. This work is purely experimental and educational. All the code shared in this repo can only be used for education purpose.

A quick summary of conclusion: most original data sample in the original dataset has a dimension of $60000\times8$. We first compress it to $10\times8$ (flatten to $1\times80$) by non-probability sampling in Phase 1. Then we apply PCA and futher reduce the dimension of each data sample to $1\times6$ and the trained logistic regression classifier achieves a best accuracy of 100%.

