#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "nn.h"

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

void mat_free(Mat m) {
  free(m.data);
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

void nn_free(NN nn) {
  for (size_t i = 0; i < nn.count; i++) {
    mat_free(nn.ws[i]);
    mat_free(nn.bs[i]);
  }
}

// Fisher-Yates shuffle
void shuffle_indices(size_t *arr, int len) {
  for (int i = len-1; i > 0 ; i--) {
    int j = rand() % (i + 1);
    size_t temp = arr[j];
    arr[j] = arr[i];
    arr[i] = temp;
  }
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

// TODO: allocate g once and pass as parameter rather than on each backprop
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
  nn_free(g);
}

void nn_learn(NN nn, Mat *xs, Mat* x_train, Mat* y_train, size_t n) {
  size_t indices[n];
  for (size_t i = 0; i < n; i++) {
    indices[i] = i;
  }
  
  for (size_t i = 0; i < EPOCHS; i++) {
    shuffle_indices(indices, n);
    
#ifdef SHOW_COST
    float cost = 0.0;
#endif
    
    for (size_t j = 0; j < n; j++) {
      //printf("training point %d\n", j);
      size_t idx = indices[j];
      nn_forward(nn, xs, x_train[idx]);

#ifdef SHOW_COST
      cost += mat_mse(xs[nn.layers - 1], y_train[idx]);
#endif
      
      nn_backprop(nn, xs, y_train[idx], RATE);
    }

#ifdef SHOW_COST
    cost /= n;
    printf("cost = %f\n", cost);
#endif
    
  }
}

float mat_mse(Mat x, Mat y) {
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

//// 1. nn_save(nn, "./mymodel.txt)
//// 2. nn_load("./mymodel.txt")
