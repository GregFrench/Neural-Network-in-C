#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist_loader.h"
#include <string.h>
#include <stdio.h>

mnist_data * mnist_loader(char * type) {
  mnist_data * data;
  unsigned int cnt;
  int ret = 1;

  if (strcmp(type, "train") == 0) {
    ret = mnist_load("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte", &data, &cnt);
  } else  if (strcmp(type, "test") == 0) {
    ret = mnist_load("./data/t10k-images.idx3-ubyte", "./data/t10k-labels.idx1-ubyte", &data, &cnt);
  }

  printf("%d\n", cnt);
  printf("%f\n", data->data[0][0]);

  if (ret == 1) {
    printf("An error occured: %d\n", ret);
  }

  return data;
}
