#include <stdio.h>
#include <stdlib.h>
#include "network.h"
#include "misc_functions.h"

int main() {
	/* int sizes[] = {784, 30, 10}; */
  	mnist_data * training_data = mnist_loader("train");
  	/*mnist_data * test_data = mnist_loader("test");*/
	int i = 0;
	int j = 0;

	for (i = 0; i < 28; i++) {
		for (j = 0; j < 28; j++) {
			printf("%f ", training_data->data[i][j]);
		}

		printf("\n");
	}

	/* printf("%f", training_data->data[1680000][60000]); */

	/*Network * net = init(sizes, 3);
	sgd(net, training_data, TRAIN_SET_SIZE, 30, 10, 100.0, test_data, TEST_SET_SIZE);*/

	free(training_data);

  	return 0;
}
