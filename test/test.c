#include <stdlib.h>

double *** set_test_biases(double biases_data[], int sizes[], int size) {
  double *** biases = malloc(sizeof(double) * (size - 1));
  int i = 0;
  int j = 0;
  int counter = 0;

  for (i = 1; i < size; i++) {
    biases[i - 1] = malloc(sizeof(double) * sizes[i]);

    for (j = 0; j < sizes[i]; j++) {
      biases[i - 1][j] = malloc(sizeof(double));
      biases[i - 1][j][0] = biases_data[counter++];
    }
  }

  /*for (i = 0; i < 2; i++) {
    biases[0][i] = malloc(sizeof(double) * 1);
  }

  biases[0][0][0] = 0.49671415;
  biases[0][1][0] = -0.1382643;
  biases[1][0][0] = 0.64768854;
  biases[1][1][0] = 1.52302986;*/

  return biases;
}

int * set_test_sizes(int sizes[], int size) {
  int i = 0;
  int * netSizes = malloc(sizeof(int) * size);

  /* copy over the sizes array into the net->sizes array */
  for (i = 0; i < size; i++) {
    netSizes[i] = sizes[i];
  }

  return netSizes;
}

double *** set_test_weights(double weights_data[], int sizes[], int size) {
  double *** weights = malloc(sizeof(double) * (size - 1));
  int i = 0;
  int j = 0;
  int k = 0;
  int counter = 0;

  for (i = 1; i < size; i++) {
    weights[i-1] = malloc(sizeof(double) * sizes[i]);

    for (j = 0; j < sizes[i]; j++) {
      weights[i-1][j] = malloc(sizeof(double) * sizes[i - 1]);

      for (k = 0; k < sizes[i - 1]; k++) {
        weights[i-1][j][k] = weights_data[counter++];
      }
    }
  }

  /*weights[0][0][0] = -0.23415337;
  weights[0][0][1] = -0.23413696;
  weights[0][0][2] = 1.57921282;
  weights[0][0][3] = 0.76743473;

  weights[0][1][0] = -0.46947439;
  weights[0][1][1] = 0.54256004;
  weights[0][1][2] = -0.46341769;
  weights[0][1][3] = -0.46572975;

  weights[1][0][0] = 0.24196227;
  weights[1][0][1] = -1.91328024;
  weights[1][1][0] = -1.72491783;
  weights[1][1][1] = -0.56228753;*/

  /*weights:
[array([[-0.23415337, -0.23413696,  1.57921282,  0.76743473],
       [-0.46947439,  0.54256004, -0.46341769, -0.46572975]]), array([[ 0.24196227, -1.91328024],
       [-1.72491783, -0.56228753]])]*/

  return weights;
}
