# Deep Learning and Optimization with JAX
Implementation of optimization algorithms with JAX and Numpy to solve deep learning problems. 

<p align='center'>
  <img width="600", img height="242", img src="https://github.com/mingboiz/jax/blob/main/img/deblurring.png">
</p>

`image_deblurring` and `image_inpainting` notebooks implements low-level solutions to Image Deblurring and Image Inpainting problems respectively without utilizing external libraries using JAX and Numpy.

<p align='center'>
  <img width="600", img height="242", img src="https://github.com/mingboiz/jax/blob/main/img/inpainting.png?raw=true">
 </p>
 
`optimization_algos` notebook implements a variety of optimization algorithms that are commonly used in Deep Learning to solve a Image Classification problem from scratch using JAX and Numpy:

- Stochastic Gradient Descent
- Stochastic Gradient Descent with Momentum + L2 Regularization
- ADAM
- L-BFGS

CelebA dataset was modified into a binary classification problem of whether an image is Male or Female - intentionally done to make this a very simplified problem as the aim is to gain deeper understanding of deep learning, optimization and its implementation with JAX and Numpy. Training is done on the first 15,000 images, testing on the last 5,000 images, and a Logistic Regression Classifier is able to achieve 95+ validation (left) and test (right) AUC 

 <p align='center'>
  <img width="508", img height="254", img src="https://github.com/mingboiz/jax/blob/main/img/img_classification.png?raw=true">
 </p>
