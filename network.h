typedef struct {
  double *** biases;
  double *** weights;
  int miniBatchSize;
  int numLayers;
  int * sizes;
  int testDataSize;
  int trainingDataSize;
} Network;

typedef struct {
  double *** nabla_b;
  double *** nabla_w;
} nabla_tuple;

#define MNIST_DATA_TYPE double

typedef struct mnist_data {
	MNIST_DATA_TYPE data[28][28]; /* 28x28 data for the image */
	unsigned int label; /* label : 0 to 9 */
} mnist_data;

mnist_data * mnist_loader(char * type);
#define TRAIN_SET_SIZE 60000
#define TEST_SET_SIZE 10000

Network * init(int sizes[], int size);
nabla_tuple * backprop(Network * net, int ** x, int ** y);
double ** cost_derivative(double ** output_activations, int ** y, int output_size);
double ** feedforward(Network * net, int a);
void free_activation(double ** activation, int num);
void free_activations(double *** activations);
void free_delta(double ** delta);
void free_double_2d(double ** arr, int num);
void free_z(double ** z);
void free_zs(double ** zs);
void free_network(Network * net);
void free_nabla_tuple(nabla_tuple * tuple);
void free_weights(double *** weights, int sizes[], int numLayers);
void sgd(Network * net, mnist_data * trainingData, int trainingDataSize, int epochs, int miniBatchSize, double eta, mnist_data * testData, int testDataSize);
void update_mini_batch(Network * net, double **** mini_batch, double eta);

/*
int evaluate(double **** test_data);
double *** setGradientVector(Network * net);
*/
