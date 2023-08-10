#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

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

#define SHOW_COST 0
#define EPOCHS 10000
#define RATE 1

typedef struct {
  Mat *ws;
  Mat *bs;
  size_t* shape;
  size_t count; // length ws, bs
  size_t layers; // length shape
} NN;

void mat_print(Mat m);
Mat mat_alloc(size_t rows, size_t cols);
NN nn_alloc(size_t nlayers, size_t* shape);
void nn_print(NN nn);
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
float mse(Mat x, Mat y);


int main() {
  srand(time(NULL));
  
  size_t shape[] = {2, 2, 1};
  NN nn = nn_alloc(3, shape);
  nn_rand(nn);

  // XOR training data
  Mat x[4], y[4];
  
  for (size_t i = 0; i < 4; i++) {
    x[i] = mat_alloc(2, 1);
    y[i] = mat_alloc(1, 1);
  }
  
  float
    x0[] = {0, 0}, y0[] = {0},
    x1[] = {1, 0}, y1[] = {1},
    x2[] = {0, 1}, y2[] = {1},
    x3[] = {1, 1}, y3[] = {0};
  
  x[0].data = x0, y[0].data = y0;
  x[1].data = x1, y[1].data = y1;
  x[2].data = x2, y[2].data = y2;
  x[3].data = x3, y[3].data = y3;

  Mat acts[3];
  acts[0] = mat_alloc(2, 1);
  acts[1] = mat_alloc(2, 1);
  acts[2] = mat_alloc(1, 1);

  for (size_t i = 0; i < 4; i++) {
    nn_forward(nn, acts, x[i]);
    Mat y_pred = acts[2];
    printf("\n");
    printf("%d ^ %d = %f", (int)x[i].data[0], (int)x[i].data[1], y_pred.data[0]);
  }

  printf("\n");
  nn_learn(nn, acts, x, y, 4);
  
  for (size_t i = 0; i < 4; i++) {
    nn_forward(nn, acts, x[i]);
    Mat y_pred = acts[2];
    printf("\n");
    printf("%d ^ %d = %f", (int)x[i].data[0], (int)x[i].data[1], y_pred.data[0]);
  }
  return 0;
}

void mat_print(Mat m) {
  printf("\n");
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      printf("%f ", ENTRY(m, i, j));
    }
    printf("\n");
  }
}

void nn_print(NN nn) {
  printf("Neural Network:\n");
  for (int i = 0; i < nn.count; i++) {
    printf("\nLayer %d\n\n", i+1);
    printf("W%d:\n", i+1);
    mat_print(nn.ws[i]);
    printf("\nB%d:\n", i+1);
    mat_print(nn.bs[i]);
  }
}

Mat mat_alloc(size_t rows, size_t cols) {
  Mat m;
  m.rows = rows;
  m.cols = cols;
  m.data = malloc(rows*cols*sizeof(float));
  assert(m.data != NULL);
  return m;
}

NN nn_alloc(size_t nlayers, size_t* shape) {
  NN nn;
  nn.layers = nlayers;
  nn.count = nlayers - 1;
  nn.shape = shape;
  nn.ws = malloc(nn.count * sizeof(Mat));
  assert(nn.ws != NULL);
  nn.bs = malloc(nn.count * sizeof(Mat));
  assert(nn.bs != NULL);
  
  
  for (size_t i = 0; i < nlayers-1; i++) {
    nn.ws[i] = mat_alloc(shape[i], shape[i+1]);
    nn.bs[i] = mat_alloc(shape[i+1], 1);
  }

  return nn;
}

float randf() {
  return  (float)rand() / (float)RAND_MAX - 0.5f;
}

void mat_rand(Mat m) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      ENTRY(m, i, j) = randf();
    }
  }
}

void nn_rand(NN nn) {
  for (size_t i = 0; i < nn.count; i++) {
    mat_rand(nn.ws[i]);
    mat_rand(nn.bs[i]);
  }
}

float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

void mat_sigmoid(Mat m) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      ENTRY(m, i, j) = sigmoid(ENTRY(m, i, j));
    }
  }
}
 
void mat_mul(Mat c, Mat a, Mat b, T_flag flag) {
  
  switch (flag){
    
  case NO_TRANS:
    assert(a.cols == b.rows);
    assert(c.rows == a.rows && c.cols == b.cols);
    for (size_t i = 0; i < c.rows; i++) {
      for (size_t j = 0; j < c.cols; j++) {
	for (size_t k = 0; k < a.cols; k++) {
	  ENTRY(c, i, j) += ENTRY(a, i, k)*ENTRY(b, k, j);
	}
      }
    }
    break;


    case TRANSA:
    assert(a.rows == b.rows);
    assert(c.rows == a.cols && c.cols == b.cols);
    for (size_t i = 0; i < c.rows; i++) {
      for (size_t j = 0; j < c.cols; j++) {
	for (size_t k = 0; k < b.rows; k++) {
	  ENTRY(c, i, j) += ENTRY(a, k, i)*ENTRY(b, k, j);
	}
      }
    }
    break;

    case TRANSB:
    assert(a.cols == b.cols);
    assert(c.rows == a.rows && c.cols == b.rows);
    for (size_t i = 0; i < c.rows; i++) {
      for (size_t j = 0; j < c.cols; j++) {
	for (size_t k = 0; k < a.cols; k++) {
	  ENTRY(c, i, j) += ENTRY(a, i, k)*ENTRY(b, j, k);
	}
      }
    }
    break;  
  }
}

void mat_scale(Mat m, float k) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      ENTRY(m, i, j) *= k;
    }
  }
}

void mat_add(Mat a, Mat b) {
  assert(a.rows == b.rows && a.cols == b.cols);
  for (size_t i = 0; i < a.rows; i++) {
    for (size_t j = 0; j < a.cols; j++) {
      ENTRY(a, i, j) += ENTRY(b, i, j);
    }
  }
}

void mat_sub(Mat a, Mat b) {
  assert(a.rows == b.rows && a.cols == b.cols);
  for (size_t i = 0; i < a.rows; i++) {
    for (size_t j = 0; j < a.cols; j++) {
      ENTRY(a, i, j) -= ENTRY(b, i, j);
    }
  }
}

void mat_copy(Mat dest, Mat src) {
  assert(dest.rows == src.rows && dest.cols == src.cols);
  for (size_t i = 0; i < dest.rows; i++) {
    for (size_t j = 0; j < dest.cols; j++) {
      ENTRY(dest, i, j) = ENTRY(src, i, j);
    }
  }
}

void nn_forward(NN nn, Mat* xs, Mat input) {
  mat_copy(xs[0], input);
  for (size_t i = 0; i < nn.count; i++) {
    mat_copy(xs[i+1], nn.bs[i]);
    mat_mul(xs[i+1], nn.ws[i], xs[i], TRANSA);
    mat_sigmoid(xs[i+1]);
  }
}

void mat_fill(Mat m, float val) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      ENTRY(m, i, j) = val;
    }
  }
}

void nn_fill(NN nn, float val) {
  for (size_t i = 0; i < nn.count; i++) {
    mat_fill(nn.ws[i], val);
    mat_fill(nn.bs[i], val);
  }
}

void nn_backprop(NN nn, Mat* xs, Mat y_truth, float rate) {
  NN g = nn_alloc(nn.layers, nn.shape);
  nn_fill(g, 0.0f);
  Mat y_pred = xs[nn.layers-1];

  float x, y;
  size_t i;
  // l must be an int since decrementing
  int l;
  // compute gradient on output layer
  // grad_b
  for (i = 0; i < g.bs[g.count-1].rows; i++) {
    x = ENTRY(y_pred, i, 0);
    y = ENTRY(y_truth, i, 0);
    ENTRY(g.bs[g.count-1], i, 0) = (x - y) * x * (1 - x);
  }
  // grad_w
  mat_mul(g.ws[g.count-1], xs[nn.count-1], g.bs[g.count-1], TRANSB);

  // compute gradient on hidden layers
  for (l = nn.count-2; l >= 0; l--) {
    // grad_b
    mat_mul(g.bs[l], nn.ws[l+1], g.bs[l+1], NO_TRANS);
    for (i = 0; i < g.bs[l].rows; i++) {
      //////// added 1 to x index //////////
      x = ENTRY(xs[l+1], i, 0);
      ENTRY(g.bs[l], i, 0) *= x * (1 - x);
    }
    // grad_w
    mat_mul(g.ws[l], xs[l], g.bs[l], TRANSB);
  }

  // update weights
  for (i = 0; i < nn.count; i++) {    
    mat_scale(g.ws[i], rate);
    mat_scale(g.bs[i], rate);
    mat_sub(nn.ws[i], g.ws[i]);
    mat_sub(nn.bs[i], g.bs[i]);
  }
}

void nn_learn(NN nn, Mat *xs, Mat* x_train, Mat* y_train, size_t n) {
  for (size_t i = 0; i < EPOCHS; i++) {

#if SHOW_COST
    float cost = 0.0;
#endif
    
    for (size_t j = 0; j < n; j++) {
      nn_forward(nn, xs, x_train[j]);

#if SHOW_COST
      cost += mse(xs[nn.layers - 1], y_train[j]);
#endif
      
      nn_backprop(nn, xs, y_train[j], RATE);
    }

#if SHOW_COST
    cost /= n;
    printf("cost = %f\n", cost);
#endif
    
  }
}

float mse(Mat x, Mat y) {
  assert(x.rows == y.rows && x.cols == y.cols);
  float c = 0.0;
  for (size_t i = 0; i < x.rows; i++) {
    for (size_t j = 0; j < x.cols; j++) {
      float diff = ENTRY(x, i, j) - ENTRY(y, i, j);
      c += diff * diff;
    }
  }
  return c / (x.rows * x.cols);
}

