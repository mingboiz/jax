# Optimization
Low level implementation of deep learning problems and algorithms with JAX. 

`image_deblurring` and `image_inpainting` notebooks are intended to demonstrate low-level implementation of image deblurring and image inpainting problem respectively without utilizing libraries with JAX.

`optimization_algos` notebook consists of building a Logistic Regression Classifier for CelebA to solve an Binary Image Classification problem whether the image is Male or Female. Training is done on the first 15,000 images, and test is done on the last 5,000 images, and CelebA dataset was modified on the dataset to make this is a very simplified problem, because the aim is intended to demonstrate low-level implementations with JAX of key optimization algorithms from scratch.

- Stochastic Gradient Descent
- Stochastic Gradient Descent with Momentum + L2 Regularization
- ADAM
- L-BFGS