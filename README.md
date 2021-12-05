# Deep Learning and Optimization with JAX
Low level implementation of deep learning problems and algorithms with JAX. 

`image_deblurring` and `image_inpainting` notebooks implements low-level solutions to Image Deblurring and Image Inpainting problems respectively without utilizing external libraries using JAX and Numpy.

`optimization_algos` notebook implements a variety of optimization algorithms used commonly in Deep Learning from scratch using JAX and Numpy to solve Image Classification problem:

- Stochastic Gradient Descent
- Stochastic Gradient Descent with Momentum + L2 Regularization
- ADAM
- L-BFGS

CelebA dataset was modified to simplify this into a binary classification problem, whether the image is Male or Female, this intentionally done to make this a very simplified problem - the aim is to demonstrate understanding of deep learning, optimization and JAX. Training is done on the first 15,000 images, and testing on the last 5,000 images, and a Logistic Regression Classifier is able to achieve 95+ AUC.
