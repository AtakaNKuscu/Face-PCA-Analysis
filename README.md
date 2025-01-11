# Face-PCA-Analysis
As my Machine Learning homework, I've asked to find 5 eigen faces and reconstruct 10 faces which is given from Yale dataset.

First I read these images as features and find covariance matrix. Afterwards I find eigenvalues and corresponding eigenvectors. I collect first 5 maximum eigenvalues and saved their corresponding eigenvectors, saved as eigenfaces.
Using these eigenfaces I reconstruct the given faces and reported the Euclidean distances from original ones.
