typedef struct {
  double *** biases;
  double *** weights;
  int miniBatchSize;
  int numLayers;
  int * sizes;
  int testDataSize;
  int trainingDataSize;
} Network;

#define MNIST_DATA_TYPE double

typedef struct mnist_data {
	MNIST_DATA_TYPE data[28][28]; /* 28x28 data for the image */
	unsigned int label; /* label : 0 to 9 */
} mnist_data;

mnist_data * mnist_loader(char * type);
#define TRAIN_SET_SIZE 60000
#define TEST_SET_SIZE 10000

Network * init(int sizes[], int size);
void sgd(Network * net, mnist_data * trainingData, int trainingDataSize, int epochs, int miniBatchSize, double eta, mnist_data * testData, int testDataSize);
void update_mini_batch(Network * net, double **** mini_batch, double eta);

/*
int evaluate(double **** test_data);
void free_network(Network * net);
double *** setGradientVector(Network * net);
*/
