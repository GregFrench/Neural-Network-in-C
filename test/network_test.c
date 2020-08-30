/* file minunit_example.c */

#include <stdio.h>
#include <stdlib.h>
#include "minunit.h"
#include "../network.h"
#include "../misc_functions.h"

int tests_run = 0;

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

int * set_sizes_test(int sizes[], int size) {
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

    for (j = 0; j < size - 1; j++) {
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

/*static char * testFeedForwardReturnsCorrectOutput() {
  Network * net = malloc(sizeof(Network));
  double ** output;
  int sizes[] = {2, 2};
  int size = 2;
  double biases[] = {
    0.49671415,
    -0.1382643
  };

  net->numLayers = size;
  net->sizes = set_sizes_test(sizes, size);
  net->biases = set_test_biases(biases, sizes, size);
  printf("%f\n", net->biases[0][0][0]);
  printf("%f\n", net->biases[0][1][0]);

  net->weights = set_weights_test(sizes, size);
  net->trainingDataSize = 0;
  net->testDataSize = 0;
  net->miniBatchSize = 0;
  
  output = feedforward(net, 2);

  mu_assert("error, feedForward doesn't return correct output for output[0][0]", is_approx(output[0][0], 0.85718348) == 1);
  mu_assert("error, feedForward doesn't return correct output for output[0][1]", is_approx(output[0][1], 0.97188062) == 1);
  mu_assert("error, feedForward doesn't return correct output for output[1][0]", is_approx(output[1][0], 0.35284178) == 1);
  mu_assert("error, feedForward doesn't return correct output for output[1][1]", is_approx(output[1][1], 0.35284928) == 1);

  free_network(net);
  return 0;
}*/

static char * testBackpropReturnsCorrectOutputForOneInputNeuronAndOneOutputNeuron() {
  Network * net = malloc(sizeof(Network));
  nabla_tuple * output;
  int sizes[] = {1, 1};
  int size = 2;
  double biases[] = {
    0.49671415
  };

  double weights[] = {
    -0.1382643
  };

  int x_data[][1] = {
    { 1 }
  };

  int y_data[][1] = {
    { 0 }
  };

  int ** x = malloc(sizeof(int *));
  x[0] = malloc(sizeof(int));
  x[0][0] = x_data[0][0];

  int ** y = malloc(sizeof(int *));
  y[0] = malloc(sizeof(int));
  y[0][0] = y_data[0][0];

  net->numLayers = size;
  net->sizes = set_sizes_test(sizes, size);
  net->biases = set_test_biases(biases, sizes, size);
  net->weights = set_test_weights(weights, sizes, size);
  net->trainingDataSize = 0;
  net->testDataSize = 0;
  net->miniBatchSize = 0;

  output = backprop(net, x, y);

  mu_assert("error, backprop doesn't return correct output for nabla_b[0][0][0]", is_approx(output->nabla_b[0][0][0], 0.14253849) == 1);
  mu_assert("error, backprop doesn't return correct output for nabla_w[0][0][0]", is_approx(output->nabla_w[0][0][0], 0.14253849) == 1);

  free_network(net);
  free_nabla_tuple(output);

  free(x[0]);
  free(x);

  free(y[0]);
  free(y);

  return 0;
}

static char * testBackpropReturnsCorrectOutput() {
  Network * net = malloc(sizeof(Network));
  nabla_tuple * output;
  int sizes[] = {4, 2, 2};
  int size = 3;
  int x_data[] = {1, 2, 3, 4};
  int y_data[] = {0, 1};
  int ** x;
  int ** y;
  char str[100];

  net->numLayers = size;
  net->sizes = set_sizes_test(sizes, size);
  /*net->biases = set_biases_test(sizes, size);
  net->weights = set_weights_test(sizes, size);*/
  net->trainingDataSize = 0;
  net->testDataSize = 0;
  net->miniBatchSize = 0;

  output = backprop(net, x, y);

  mu_assert("error, backprop doesn't return correct output for nabla_b[0][0][0]", is_approx(output->nabla_b[0][0][0], 0.00013637) == 1);
  mu_assert("error, backprop doesn't return correct output for nabla_b[0][1][0]", is_approx(output->nabla_b[0][1][0], -0.01133838) == 1);
  mu_assert("error, backprop doesn't return correct output for nabla_b[1][0][0]", is_approx(output->nabla_b[1][0][0], 0.14780199) == 1);
  mu_assert("error, backprop doesn't return correct output for nabla_b[1][1][0]", is_approx(output->nabla_b[1][1][0], -0.13766564) == 1);

  mu_assert("error, backprop doesn't return correct output for nabla_w[0][0][0]", is_approx(output->nabla_w[0][0][0], 0.00013637) == 1);
  mu_assert("error, backprop doesn't return correct output for nabla_w[0][0][1]", is_approx(output->nabla_w[0][0][1], 0.00027274) == 1);
  mu_assert("error, backprop doesn't return correct output for nabla_w[0][0][2]", is_approx(output->nabla_w[0][0][2], 0.00040912) == 1);
  mu_assert("error, backprop doesn't return correct output for nabla_w[0][0][3]", is_approx(output->nabla_w[0][0][3], 0.00054549) == 1);

  mu_assert("error, backprop doesn't return correct output for nabla_w[0][1][0]", is_approx(output->nabla_w[0][1][0], -0.01133838) == 1);
  mu_assert("error, backprop doesn't return correct output for nabla_w[0][1][1]", is_approx(output->nabla_w[0][1][1], -0.02267675) == 1);
  mu_assert("error, backprop doesn't return correct output for nabla_w[0][1][2]", is_approx(output->nabla_w[0][1][2], -0.03401513) == 1);
  mu_assert("error, backprop doesn't return correct output for nabla_w[0][1][3]", is_approx(output->nabla_w[0][1][3], -0.0453535) == 1);

  mu_assert("error, backprop doesn't return correct output for nabla_w[1][0][0]", is_approx(output->nabla_w[1][0][0], 0.14772818) == 1);
  mu_assert("error, backprop doesn't return correct output for nabla_w[1][0][1]", is_approx(output->nabla_w[1][0][1], 0.00866807) == 1);
  mu_assert("error, backprop doesn't return correct output for nabla_w[1][1][0]", is_approx(output->nabla_w[1][1][0], -0.13759689) == 1);
  mu_assert("error, backprop doesn't return correct output for nabla_w[1][1][1]", is_approx(output->nabla_w[1][1][1], -0.00807361) == 1);

  free_network(net);
  free_nabla_tuple(output);

  /*free(x);
  free(y);*/
  return 0;
}

static char * all_tests() {
  /*mu_run_test(testFeedForwardReturnsCorrectOutput);*/
  mu_run_test(testBackpropReturnsCorrectOutputForOneInputNeuronAndOneOutputNeuron);
  /*mu_run_test(testBackpropReturnsCorrectOutput);*/

  return 0;
}

int main(int argc, char **argv) {
    char *result = all_tests();
    if (result != 0) {
        printf("%s\n", result);
    }
    else {
        printf("ALL TESTS PASSED\n");
    }
    printf("Tests run: %d\n", tests_run);

    return result != 0;
}
