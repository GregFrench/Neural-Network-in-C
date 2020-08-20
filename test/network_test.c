/* file minunit_example.c */

#include <stdio.h>
#include <stdlib.h>
#include "minunit.h"
#include "../network.h"
#include "../misc_functions.h"

int tests_run = 0;

int foo = 7;
int bar = 4;

double *** set_biases_test(int sizes[], int size) {
  double *** biases = malloc(sizeof(double) * (size - 1));
  int i = 0;

  biases[0] = malloc(sizeof(double) * 2);

  for (i = 0; i < 2; i++) {
    biases[0][i] = malloc(sizeof(double) * 1);
  }
  
  biases[0][0][0] = 0.49671415;
  biases[0][1][0] = -0.1382643;

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

double *** set_weights_test(int sizes[], int size) {
  double *** weights = malloc(sizeof(double) * (size - 1));
  int i = 0;

  weights[0] = malloc(sizeof(double) * 4);

  for (i = 0; i < 2; i++) {
    weights[0][i] = malloc(sizeof(double) * 2);
  }

  weights[0][0][0] = 0.64768854;
  weights[0][0][1] = 1.52302986;
  weights[0][1][0] = -0.23415337;
  weights[0][1][1] = -0.23413696;

  return weights;
}

static char * testFeedForwardReturnsCorrectOutput() {
  Network * net = malloc(sizeof(Network));
  double ** output;
  int sizes[] = {2, 2};
  int size = 2;

  net->numLayers = size;
  net->sizes = set_sizes_test(sizes, size);
  net->biases = set_biases_test(sizes, size);
  net->weights = set_weights_test(sizes, size);
  net->trainingDataSize = 0;
  net->testDataSize = 0;
  net->miniBatchSize = 0;
  
  output = feedforward(net, 2);

  mu_assert("error, feedForward doesn't return correct output for output[0][0]", is_approx(output[0][0], 0.85718348) == 1);
  mu_assert("error, feedForward doesn't return correct output for output[0][1]", is_approx(output[0][1], 0.97188062) == 1);
  mu_assert("error, feedForward doesn't return correct output for output[1][0]", is_approx(output[1][0], 0.35284178) == 1);
  mu_assert("error, feedForward doesn't return correct output for output[1][1]", is_approx(output[1][1], 0.35284928) == 1);
  return 0;
}

static char * all_tests() {
  mu_run_test(testFeedForwardReturnsCorrectOutput);
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
