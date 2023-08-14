#include <stdio.h>
#include <assert.h>
#include "nn.h"

unsigned int swap_endian(unsigned int n);
void mnist_print(float* img_data);
void mnist_read_img(float* img_data, FILE* fp);
size_t argmax(float* arr, size_t len);

int main(void) {
  FILE* fp;
  char buffer[4];
  
  // ---- Image File ----
  fp = fopen("./mnist/train-images.idx3-ubyte", "rb");
  assert(fp != NULL);
  
  // magic number (2051)
  fread(buffer, 1, 4, fp);
  unsigned int magic_number = swap_endian(* ((unsigned int*) buffer));
  printf("\n\n %u \n", magic_number);

  // number of images
  fread(buffer, 1, 4, fp);
  unsigned int nimages = swap_endian(* ((unsigned int*) buffer));
  printf("\n\n %u \n", nimages);

  // number of columns
  fread(buffer, 1, 4, fp);
  unsigned int nrows = swap_endian(* ((unsigned int*) buffer));
  printf("\n\n %u \n", nrows);

  // number of rows
  fread(buffer, 1, 4, fp);
  unsigned int ncols = swap_endian(* ((unsigned int*) buffer));
  printf("\n\n %u \n", ncols);


  Mat imgs[60000];
  for (size_t i = 0; i < 60000; i++) {
    imgs[i] = mat_alloc(784, 1);
    mnist_read_img(imgs[i].data, fp);
  }
  
  fclose(fp);
  
  // ---- Label File ----
  fp = fopen("./mnist/train-labels.idx1-ubyte", "rb");
  assert(fp != NULL);

  // magic number  
  fread(buffer, 1, 4, fp);
  magic_number = swap_endian(* ((unsigned int*) buffer));
  printf("\n\n %u \n", magic_number);

  // number of labels
  fread(buffer, 1, 4, fp);
  unsigned int nlabels = swap_endian(* ((unsigned int*) buffer));
  printf("\n\n %u \n", nlabels);

  unsigned char label_buffer[60000];
  fread(label_buffer, 1, 60000, fp);

  Mat labels[60000];
  for (size_t i = 0; i < 60000; i++) {
    labels[i] = mat_alloc(10, 1);
    mat_fill(labels[i], 0.0);
    ENTRY(labels[i], (size_t) label_buffer[i], 0) = 1.0;
  }
  
  fclose(fp);

  //void nn_learn(NN nn, Mat *xs, Mat* x_train, Mat* y_train, size_t n)

  size_t shape[3] = {784, 25, 10};
  NN nn = nn_alloc(3, shape);
  nn_rand(nn);
  Mat acts[] = {
    mat_alloc(784, 1),
    mat_alloc(25, 1),
    mat_alloc(10, 1)
  };
  
  nn_learn(nn, acts, imgs, labels, 60000);


  // ---- Test Image File ----
  fp = fopen("./mnist/t10k-images.idx3-ubyte", "rb");
  assert(fp != NULL);
  
  // magic number (2051)
  fread(buffer, 1, 4, fp);
  magic_number = swap_endian(* ((unsigned int*) buffer));
  printf("\n\n %u \n", magic_number);

  // number of images
  fread(buffer, 1, 4, fp);
  nimages = swap_endian(* ((unsigned int*) buffer));
  printf("\n\n %u \n", nimages);

  // number of columns
  fread(buffer, 1, 4, fp);
  nrows = swap_endian(* ((unsigned int*) buffer));
  printf("\n\n %u \n", nrows);

  // number of rows
  fread(buffer, 1, 4, fp);
  ncols = swap_endian(* ((unsigned int*) buffer));
  printf("\n\n %u \n", ncols);


  Mat test_imgs[10000];
  for (size_t i = 0; i < 10000; i++) {
    test_imgs[i] = mat_alloc(784, 1);
    mnist_read_img(test_imgs[i].data, fp);
  }
  
  fclose(fp);
  
  // ---- Test Label File ----
  fp = fopen("./mnist/t10k-labels.idx1-ubyte", "rb");
  assert(fp != NULL);

  // magic number  
  fread(buffer, 1, 4, fp);
  magic_number = swap_endian(* ((unsigned int*) buffer));
  printf("\n\n %u \n", magic_number);

  // number of labels
  fread(buffer, 1, 4, fp);
  nlabels = swap_endian(* ((unsigned int*) buffer));
  printf("\n\n %u \n", nlabels);

  unsigned char test_label_buffer[10000];
  fread(test_label_buffer, 1, 10000, fp);

  Mat test_labels[10000];
  for (size_t i = 0; i < 10000; i++) {
    test_labels[i] = mat_alloc(10, 1);
    mat_fill(test_labels[i], 0.0);
    ENTRY(test_labels[i], (size_t) test_label_buffer[i], 0) = 1.0;
  }
  
  fclose(fp);

  // Performance on test
  Mat output;
  size_t predicted_digit, actual_digit, ncorrect=0;
  for (size_t i = 0; i < 10000; i++) {
    nn_forward(nn, acts, test_imgs[i]);
    output  = acts[nn.layers-1];
    predicted_digit = argmax(output.data, 10);
    actual_digit = argmax(test_labels[i].data, 10);
    if (predicted_digit == actual_digit) ncorrect++;
  }
  printf("\n%zu out of 10000 correctly classified\n", ncorrect);
  return 0;
}

unsigned int swap_endian(unsigned int n) {
  return
    ((n & 0x000000FF) << 24)
    | ((n & 0x0000FF00) << 8)
    | ((n & 0x00FF0000) >> 8)
    | ((n & 0xFF000000) >> 24);
}

void mnist_print(float* img_data) {
  for (size_t i = 0; i < 28; i++) {
    for (size_t j = 0; j < 28; j++) {
      if (img_data[i*28 + j] < 0.33) {
	printf("###");
      } else {
	printf("   ");
      }
    }
    printf("\n");
  }
}

void mnist_read_img(float* img_data, FILE* fp) {
  unsigned char img_buffer[784];
  fread(img_buffer, 1, 784, fp);

  // normalize values to range [0, 1)
  for (size_t i = 0; i < 784; i++) {
    img_data[i] = (float) img_buffer[i] / 255.0f;
  }
}

size_t argmax(float* arr, size_t len) {
  size_t result = 0;
  for (size_t i = 1; i < len; i++) {
    if (arr[i] > arr[result]) result = i;
  }
  return result;
}






