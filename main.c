#include <stdio.h>
#include <stdlib.h>
#include "network.h"
#include "misc_functions.h"

double *** set_biases_test(int sizes[], int size) {
  double *** biases = malloc(sizeof(double) * (size - 1));
  int i = 0;

  biases[0] = malloc(sizeof(double) * 2);

  for (i = 0; i < 2; i++) {
    biases[0][i] = malloc(sizeof(double) * 2);
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

int main() {
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

	printf("%f\n", output[0][0]);
	printf("%f\n", output[0][1]);
	printf("%f\n", output[1][0]);
	printf("%f\n", output[1][1]);

  	return 0;
}
