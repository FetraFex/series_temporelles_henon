# Methodology

## Hénon map

The Hénon system is a discrete dynamical system defined by:

```
x_{n+1} = 1 - a * x_n^2 + y_n
y_{n+1} = b * x_n
```

with classic chaotic parameters a = 1.4, b = 0.3.

## Takens' embedding theorem

A scalar time series can be reconstructed into a state space using delay
coordinates:

```
X_n = [x_n, x_{n-tau}, ..., x_{n-(d-1)*tau}]
```

The embedding dimension `d` and delay `tau` recover the topology of the
original attractor.

## From-scratch MLP

A Multilayer Perceptron is implemented manually: neurons, layers, weights,
biases, activation functions, forward propagation, loss, backpropagation, and
gradient-based weight updates — without any deep-learning framework.
