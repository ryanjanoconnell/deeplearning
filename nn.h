#ifndef NN_H
#define NN_H

#include <stddef.h>

typedef struct {
  size_t rows;
  size_t cols;
  float *data;
} Mat;

typedef enum {
  NO_TRANS,
  TRANSA,
  TRANSB
} T_flag;

#define ENTRY(m, i, j) (m).data[(i) * (m).cols + (j)]

#ifndef EPOCHS
#define EPOCHS 10000
#endif

#ifndef RATE
#define RATE 1
#endif

typedef struct {
  Mat *ws;
  Mat *bs;
  size_t* shape;
  size_t count; // length ws, bs
  size_t layers; // length shape
} NN;

void mat_print(Mat m);
Mat mat_alloc(size_t rows, size_t cols);
void mat_free(Mat m);
NN nn_alloc(size_t nlayers, size_t* shape);
void nn_free(NN nn);
void nn_print(NN nn);
void shuffle_indices(size_t *arr, int len);
float randf();
void mat_rand(Mat m);
void nn_rand(NN nn);
float sigmoid(float x);
void mat_sigmoid(Mat m);
void mat_mul(Mat c, Mat a, Mat b, T_flag flag);
void mat_scale(Mat m, float k);
void mat_add(Mat a, Mat b);
void mat_sub(Mat a, Mat b);
void mat_copy(Mat dest, Mat src);
void nn_forward(NN nn, Mat* xs, Mat input);
void mat_fill(Mat m, float val);
void nn_fill(NN nn, float val);
void nn_backprop(NN nn, Mat* xs, Mat y_truth, float rate);
void nn_learn(NN nn, Mat *xs, Mat* x_train, Mat* y_train, size_t n);
float mat_mse(Mat x, Mat y);

#endif // NN_H
