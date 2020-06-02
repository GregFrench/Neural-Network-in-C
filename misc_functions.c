#include <math.h>

/* The sigmoid function */
double sigmoid(double z) {
  return 1 / (1 + exp(-z));
}

/* Derivative of the sigmoid function */
double sigmoid_prime(double z) {
  return sigmoid(z) * (1 - sigmoid(z));
}
