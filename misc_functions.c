#include <stdlib.h>
#include <math.h>
#include "misc_functions.h"

/* samples values from the gaussian distribution with a mean of 0 and variance of 1 */
double box_muller() {
  double u1 = 0.00;
  double u2 = 0.00;
  double n1 = 0.00;
  double epsilon = 1.0;

  /* get the epsilon value by calculating machine precision */
  while (epsilon + 1 > 1) {
    epsilon /= 2;
  }

  epsilon *= 2;

  while (u1 < epsilon || u2 < epsilon) {
    u1 = rand_double(0.0, 1.0);
    u2 = rand_double(0.0, 1.0);
  }

  n1 = sqrt(-2 * log(u1)) * cos(2 * PI * u2);

  return n1;
}

double ** create_double_matrix(int m, int n) {
  int i, j = 0;
  double ** matrix = malloc(sizeof(double *) * m);

  for (i = 0; i < m; i++) {
    matrix[i] = malloc(sizeof(double) * n);

    for (j = 0; j < n; j++) {
      matrix[i][j] = 0.00;
    }
  }

  return matrix;
}

int is_approx(double x, double y) {
  return fabs(x - y) < EPSILON;
}

double ** dot2D(double ** matrix, double ** matrix2, int m, int n) {
  double ** result = malloc(sizeof(double *) * m);
  int i = 0;
  int j = 0;
  double sum = 0;

  for (i = 0; i < m; i++) {
    sum = 0;
    result[i] = malloc(sizeof(double) * n);

    for (j = 0; j < n; j++) {
      sum += (matrix[i][j] * matrix2[j][0]);
    }

    result[i][0] = sum;
  }

  return result;
}

double ** dotMatrixByScalar(double ** matrix, int scalar) {
  double ** result = malloc(sizeof(double) * 2);
  int i = 0;
  int j = 0;

  for (i = 0; i < 2; i++) {
    result[i] = malloc(sizeof(double) * 2);

    for (j = 0; j < 2; j++) {
      result[i][j] = matrix[i][j] * scalar;
    }
  }

  return result;
}

/* Similar to the numpy np.random.randn(x, y) function using two parameters */
double ** randn(int x, int y) {
  int i = 0;
  int j = 0;

  double ** random = malloc(sizeof(double) * x);

  for (i = 0; i < x; i++) {
    random[i] = malloc(sizeof(double) * y);

    for (j = 0; j < y; j++) {
      random[i][j] = box_muller();
    }
  }

  return random;
}

/* randomly generate a double value between a and b inclusive */
double rand_double(double a, double b) {
  int temp = 0;
  double random_num;

  /* swap b and a if a greater than b */
  if (b < a) {
    temp = a;
    a = b;
    b = temp;
  }

  random_num = ((double) rand() / (double) RAND_MAX) * (b - a) + a;

  return random_num;
}

double *** set_biases(int sizes[], int size) {
  double *** biases = malloc(sizeof(double) * (size - 1));
  int i = 0;

  for (i = 1; i < size; i++) {
    biases[i-1] = randn(sizes[i], 1);
  }

  return biases;
}

int * set_sizes(int sizes[], int size) {
  int i = 0;
  int * netSizes = malloc(sizeof(int) * size);

  /* copy over the sizes array into the net->sizes array */
  for (i = 0; i < size; i++) {
    netSizes[i] = sizes[i];
  }

  return netSizes;
}

double *** set_weights(int sizes[], int size) {
  double *** weights = malloc(sizeof(double) * (size - 1));
  int i = 0;

  for (i = 1; i < size; i++) {
    weights[i-1] = randn(sizes[i-1], sizes[i]);
  }

  return weights;
}

void shuffle(double **** array, size_t n) {
  size_t i;
  double *** t;

  if (n > 1) {
    for (i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

/* The sigmoid function */
double sigmoid(double z) {
  return 1 / (1 + exp(-z));
}

/* Derivative of the sigmoid function */
double ** sigmoid_prime(double ** z, int m) {
  int i = 0;
  double ** arr = malloc(sizeof(double *) * m);

  for (i = 0; i < m; i++) {
    arr[i] = malloc(sizeof(double));
    arr[i][0] = sigmoid(z[i][0]) * (1 - sigmoid(z[i][0]));
  }

  return arr;
}

double ** transpose(double ** matrix, int m, int n) {
  int i, j = 0;
  double ** result = create_double_matrix(n, m);

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      result[j][i] = matrix[i][j];
    }
  }

  return result;
}
