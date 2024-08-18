## Fisher LDA (Fisher Linear Discriminant Analysis)

In this project, we will apply a Fisher LDA from a set of images of faces, each of size 256*256 in RGB format.

**Dataset** : *jaffe*

We reconstruct the original data by using K basis vectors obtained from LDA.

Below images show reconstructed images of one person for k=1, 6, 29):

![reconstructed images1](https://github.com/Ghafarian-code/Fisher-LDA/blob/master/images/Figure_1.png)
![reconstructed images2](https://github.com/Ghafarian-code/Fisher-LDA/blob/master/images/Figure_2.png)
![reconstructed images3](https://github.com/Ghafarian-code/Fisher-LDA/blob/master/images/Figure_3.png)
![reconstructed images4](https://github.com/Ghafarian-code/Fisher-LDA/blob/master/images/Figure_4.png)
![reconstructed images5](https://github.com/Ghafarian-code/Fisher-LDA/blob/master/images/Figure_5.png)

Plot the MSE between the original and reconstructed images in terms of number of eigenvectors:
![MSE plot](https://github.com/Ghafarian-code/Fisher-LDA/blob/master/images/Figure_6.png)

> The problem of applying Fisher LDA on the dataset is that calculating the inverse of the covariance matrix of classes in high-dimensional data sets is very expensive. 
Also, if we have a large number of outliers in the dataset, A lot of noise is added to the average data, so lda cannot perform as expected.
