/* file minunit_example.c */

#include <stdio.h>
#include <stdlib.h>
#include "minunit.h"
#include "test.h"
#include "../network.h"
#include "../misc_functions.h"

int tests_run = 0;

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
  net->sizes = set_test_sizes(sizes, size);
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

static char * testBackpropReturnsCorrectOutputForOneInputNeuronOneHiddenNeuronAndOneOutputNeuron() {
  /*Network * net = malloc(sizeof(Network));
  nabla_tuple * output;
  int sizes[] = {1, 1, 1};
  int size = 3;
  double biases[] = {
    0.49671415,
    -0.1382643
  };

  double weights[] = {
    0.64768854,
    1.52302986
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
  net->sizes = set_test_sizes(sizes, size);
  net->biases = set_test_biases(biases, sizes, size);
  net->weights = set_test_weights(weights, sizes, size);
  net->trainingDataSize = 0;
  net->testDataSize = 0;
  net->miniBatchSize = 0;

  output = backprop(net, x, y);

  mu_assert("backprop test #2: error, backprop doesn't return correct output for nabla_b[0][0][0]", is_approx(output->nabla_b[0][0][0], 0.03996702) == 1);
  mu_assert("backprop test #2: error, backprop doesn't return correct output for nabla_b[1][0][0]", is_approx(output->nabla_b[1][0][0], 0.14325333) == 1);
  mu_assert("backprop test #2: error, backprop doesn't return correct output for nabla_w[0][0][0]", is_approx(output->nabla_w[0][0][0], 0.03996702) == 1);
  mu_assert("backprop test #2: error, backprop doesn't return correct output for nabla_w[1][0][0]", is_approx(output->nabla_w[1][0][0], 0.1086558) == 1);

  free_network(net);
  free_nabla_tuple(output);

  free(x[0]);
  free(x);

  free(y[0]);
  free(y);*/

  return 0;
}

static char * testCostDerivativeReturnsCorrectOutputForOneOutputNeuron() {
  double ** output;
  int num_output_neurons = 1;

  double activation_data[][1] = {
    { 0.73437498 }
  };

  int y_data[][1] = {
    { 0 }
  };

  double ** activations = malloc(sizeof(double *));
  activations[0] = malloc(sizeof(double));
  activations[0][0] = activation_data[0][0];

  int ** y = malloc(sizeof(int *));
  y[0] = malloc(sizeof(int));
  y[0][0] = y_data[0][0];

  output = cost_derivative(activations, y, num_output_neurons);

  mu_assert("cost_derivative test #1: error, cost_derivative doesn't return correct output for output[0][0]", is_approx(output[0][0], 0.73437498) == 1);

  free(output[0]);
  free(output);

  free(activations[0]);
  free(activations);

  free(y[0]);
  free(y);

  return 0;
}

static char * testCostDerivativeReturnsCorrectOutputForTwoOutputNeurons() {
  double ** output;
  int num_output_neurons = 2;

  double activation_data[][1] = {
    { 0.9850865 },
    { 0.30138915 }
  };

  int y_data[][1] = {
    { 0 },
    { 1 }
  };

  double ** activations = malloc(sizeof(double *));
  activations[0] = malloc(sizeof(double));
  activations[1] = malloc(sizeof(double));
  activations[0][0] = activation_data[0][0];
  activations[1][0] = activation_data[1][0];

  int ** y = malloc(sizeof(int *));
  y[0] = malloc(sizeof(int));
  y[1] = malloc(sizeof(int));
  y[0][0] = y_data[0][0];
  y[1][0] = y_data[1][0];

  output = cost_derivative(activations, y, num_output_neurons);

  mu_assert("cost_derivative test #2: error, cost_derivative doesn't return correct output for output[0][0]", is_approx(output[0][0], 0.9850865) == 1);
  mu_assert("cost_derivative test #2: error, cost_derivative doesn't return correct output for output[1][0]", is_approx(output[1][0], -0.69861085) == 1);

  free_double_2d(output, num_output_neurons);
  free_double_2d(activations, num_output_neurons);
  free_int_2d(y, num_output_neurons);

  return 0;
}

static char * testBackpropReturnsCorrectOutput() {
  /*Network * net = malloc(sizeof(Network));
  nabla_tuple * output;
  int sizes[] = {4, 2, 2};
  int size = 3;
  int x_data[] = {1, 2, 3, 4};
  int y_data[] = {0, 1};
  int ** x;
  int ** y;

  net->numLayers = size;
  net->sizes = set_test_sizes(sizes, size);
  net->biases = set_biases_test(sizes, size);
  net->weights = set_weights_test(sizes, size);
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

  free(x);
  free(y);*/
  return 0;
}

static char * all_tests() {
  /*mu_run_test(testFeedForwardReturnsCorrectOutput);*/
  mu_run_test(testBackpropReturnsCorrectOutputForOneInputNeuronAndOneOutputNeuron);
  mu_run_test(testBackpropReturnsCorrectOutputForOneInputNeuronOneHiddenNeuronAndOneOutputNeuron);
  mu_run_test(testCostDerivativeReturnsCorrectOutputForOneOutputNeuron);
  mu_run_test(testCostDerivativeReturnsCorrectOutputForTwoOutputNeurons);

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
