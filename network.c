#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "misc_functions.h"
#include "network.h"

Network * init(int sizes[], int size) {
  Network * net = malloc(sizeof(Network));

  net->numLayers = size;
  net->sizes = set_sizes(sizes, size);
  net->biases = set_biases(sizes, size);
  net->weights = set_weights(sizes, size);
  net->trainingDataSize = 0;
  net->testDataSize = 0;
  net->miniBatchSize = 0;

  return net;
}

nabla_tuple * backprop(Network * net, int ** x, int ** y) {
  nabla_tuple * tuple = malloc(sizeof(nabla_tuple));
  int i = 0;
  int j = 0;
  int k = 0;

  double ** activation = malloc(sizeof(double *));
  activation[0] = malloc(sizeof(double));

  double *** activations = malloc(sizeof(double *) * (net->numLayers - 1));
  double ** z;
  double ** zs = malloc(sizeof(int *));
  zs[0] = malloc(sizeof(double));
  double ** delta = malloc(sizeof(double *));
  delta[0] = malloc(sizeof(double));

  tuple->nabla_b = malloc(sizeof(double *) * 1);

  for (i = 0; i < 1; i++) {
    tuple->nabla_b[i] = malloc(sizeof(double *) * 1);

    for (j = 0; j < 1; j++) {
      tuple->nabla_b[i][j] = malloc(sizeof(double) * 1);

      for (k = 0; k < 1; k++) {
        tuple->nabla_b[i][j][k] = 0.00;
      }
    }
  }

  tuple->nabla_w = malloc(sizeof(double *) * 1);

  for (i = 0; i < 1; i++) {
    tuple->nabla_w[i] = malloc(sizeof(double *) * 1);

    for (j = 0; j < 1; j++) {
      tuple->nabla_w[i][j] = malloc(sizeof(double) * 1);

      for (k = 0; k < 1; k++) {
        tuple->nabla_w[i][j][k] = 0.00;
      }
    }
  }

  activation[0][0] = x[0][0];

  for (i = 0; i < net->numLayers - 1; i++) {
    z = dot2D(net->weights[i], activation, 1, 1);
    z[0][0] += net->biases[i][0][0];
    zs[0][0] = z[0][0];
    activation[0][0] = sigmoid(z[0][0]);
    activations[i] = activation;

    printf("activations: %f\n", activations[i][0][0]);
  }

  delta[0][0] = (activations[0][0][0] - y[0][0]) * sigmoid_prime(zs[0][0]);
  tuple->nabla_b[0][0][0] = delta[0][0];
  tuple->nabla_w[0][0][0] = delta[0][0];

  free_activation(activation, 1);
  free(activations);
  free_z(z);
  free_zs(zs);
  free_delta(delta);

  return tuple;
}

void free_activation(double ** activation, int num) {
  int i = 0;

  for (i = 0; i < num; i++) {
    free(activation[i]);
  }

  free(activation);
}

void free_activations(double *** activations) {
  free(activations[0][0]);
  free(activations[0]);
  free(activations);
}

void free_delta(double ** delta) {
  free(delta[0]);
  free(delta);
}

void free_z(double ** z) {
  free(z[0]);
  free(z);
}

void free_zs(double ** zs) {
  free(zs[0]);
  free(zs);
}

double ** cost_derivative(double ** output_activations, int ** y, int output_size) {
  double ** cost = malloc(sizeof(double *) * output_size);
  int i = 0;

  for (i = 0; i < output_size; i++) {
    cost[i] = malloc(sizeof(double));
    cost[i][0] = output_activations[i][0] - y[i][0];
  }

  return cost;
}

/* int evaluate(double **** test_data) {
  test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data];
  return sum(int(x == y) for (x, y) in test_results);

  return 0;
} */

double ** feedforward(Network * net, int a) {
  double ** result = malloc(sizeof(double) * 4);
  double ** weights = net->weights[0];
  double ** biases = net->biases[0];
  double ** dot = dotMatrixByScalar(weights, a);
  int i = 0;
  int j = 0;

  result[0] = malloc(sizeof(double) * 2);
  result[1] = malloc(sizeof(double) * 2);

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
      result[i][j] = sigmoid(dot[i][j] + biases[i][0]);
    }
  }

  return result;
}

void free_double_2d(double ** arr, int num) {
  int i = 0;

  for (i = 0; i < num; i++) {
    free(arr[i]);
  }

  free(arr);
}

void free_network(Network * net) {
  int i = 0;
  int j = 0;

  for (i = 1; i < net->numLayers; i++) {
    for (j = 0; j < net->sizes[i]; j++) {
      free(net->biases[i - 1][j]);
    }

    free(net->biases[i - 1]);
  }

  free(net->biases);

  free_weights(net->weights, net->sizes, net->numLayers);

  free(net->sizes);

  free(net);
}

void free_nabla_tuple(nabla_tuple * tuple) {
  int i = 0;
  int j = 0;

  for (i = 0; i < 1; i++) {
    for (j = 0; j < 1; j++) {
      free(tuple->nabla_b[i][j]);
    }

    free(tuple->nabla_b[i]);
  }

  free(tuple->nabla_b);

  for (i = 0; i < 1; i++) {
    for (j = 0; j < 1; j++) {
      free(tuple->nabla_w[i][j]);
    }

    free(tuple->nabla_w[i]);
  }

  free(tuple->nabla_w);

  free(tuple);
}

void free_weights(double *** weights, int sizes[], int numLayers) {
  int i, j = 0;

  for (i = 1; i < numLayers; i++) {
    for (j = 0; j < sizes[i]; j++) {
      free(weights[i-1][j]);
    }

    free(weights[i - 1]);
  }

  free(weights);
}

/* double *** setGradientVector(Network * net) {
  int i = 0;
  int j = 0;
  double *** nabla = NULL;

  nabla = malloc(sizeof(double) * (net->numLayers - 1));

  for (i = 0; i < net->numLayers; i++) {
    nabla[i] = malloc(sizeof(double) * net->sizes[i]);

    for (j = 0; j < net->sizes[i]; j++) {
      nabla[i][j] = malloc(sizeof(double));
      nabla[i][j][0] = 0.0;
    }
  }

  return nabla;
} */

void sgd(Network * net, mnist_data * trainingData, int trainingDataSize, int epochs, int miniBatchSize, double eta, mnist_data * testData, int testDataSize) {
  /*int n = trainingDataSize;
  int nTest = testDataSize;
  int numMiniBatches = 0;
  double ***** miniBatches;
  int i = 0;
  int j = 0;
  int k = 0;

  net->trainingDataSize = n;
  net->testDataSize = nTest;
  net->miniBatchSize = miniBatchSize;

  numMiniBatches = (int) ceil((double) trainingDataSize / (double) miniBatchSize);

  for (i = 0; i < epochs; i++) {
    shuffle(trainingData, trainingDataSize);

    miniBatches = malloc(sizeof(double) * numMiniBatches);

    for (j = 0; j < numMiniBatches; j++) {
      miniBatches[j] = malloc(sizeof(double) * miniBatchSize);

      for (k = 0; k < miniBatchSize; k++) {
        miniBatches[j][k] = trainingData[k];
      }
    }

    for (j = 0; j < 1; j++) {
      update_mini_batch(net, miniBatches[j], eta);
    }

    if (nTest > 0) {
      printf("Epoch %d: %d / %d", j, evaluate(testData), nTest);
    } else {
      printf("Epoch %d complete", j);
    }
  }*/
}

void update_mini_batch(Network * net, double **** miniBatch, double eta) {
  /*double *** nablaB = NULL;
  double *** nablaW = NULL;
  double *** x = NULL;
  double *** y = NULL;
  int i = 0;
  int j = 0;

  nablaB = setGradientVector(net);
  nablaW = setGradientVector(net);

  for (i = 0; i < net->miniBatchSize; i++) {
    x = miniBatch[0];
    y = miniBatch[0];
  }

  for x, y in miniBatch {
    delta_nabla_b, delta_nabla_w = backprop(x, y);
    nablaB = [nb+dnb for nb, dnb in zip(nablaB, delta_nabla_b)];
    nablaW = [nw+dnw for nw, dnw in zip(nablaW, delta_nabla_w)];
  }

  net->weights = [w-(eta/len(miniBatch))*nw for w, nw in zip(net->weights, nablaW)]
  net->biases = [b-(eta/len(miniBatch))*nb for b, nb in zip(net->biases, nablaB)]*/

  return;
}
