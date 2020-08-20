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

/* double **** backprop(Network * net, x, y) {
  nabla_b = [np.zeros(b.shape) for b in net->biases];
  nabla_w = [np.zeros(w.shape) for w in net->weights];

  activation = x;
  activations = [x];
  zs = [];

  for b, w in zip(net->biases, net->weights) {
      z = np.dot(w, activation)+b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation);
  }

  delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]);
  nabla_b[-1] = delta;
  nabla_w[-1] = dot(delta, transpose(activations[-2]));

  for l in xrange(2, net->numLayers) {
    z = zs[-l];
    sp = sigmoid_prime(z);
    delta = np.dot(net->weights[-l+1].transpose(), delta) * sp;
    nabla_b[-l] = delta;
    nabla_w[-l] = np.dot(delta, activations[-l-1].transpose());
  }

  return (nabla_b, nabla_w);
} */

/* double **** cost_derivative(output_activations, y) {
  return (output_activations-y);
} */

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

/* void free_network(Network * net) {
  int i = 0;
  int j = 0;

  free(net->sizes);

  for (i = 0; i < 1; i++) {
    for (j = 0; j < 1; j++) {
      free(net->biases[i][j]);
    }

    free(net->biases[i]);
  }

  free(net->biases);

  for (i = 0; i < 1; i++) {
    for (j = 0; j < 1; j++) {
      free(net->weights[i][j]);
    }

    free(net->weights[i]);
  }

  free(net->weights);

  free(net);
} */

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
