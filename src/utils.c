#include <stdlib.h>
#include <math.h>
#include <Arduino.h>

// Calculate the mean squared error between two arrays
float calc_mse(const float *x, const float *x_hat, const int len) {
  
  float mse = 0;

  // Square difference between each set of elements
  for (int i = 0; i < len; i++) {
    mse += pow(x[i] - x_hat[i], 2);
  }

  return mse / len;
}

// Calculate the mean of the values
float calculateMean(const float *data, size_t size) {
    float sum = 0.0, d1;

    for (int i = 0; i < size; i++) {
      // Data 1
      if(isnan(*(data + i))) {
          d1 = 0;
      }
      else {
          d1 = *(data + i);
      }
      sum += d1;
    }

    return sum / size;
}

// Calculate the standard desviation of the values
float calculateStd(const float *data, size_t size) {
    float diff = 0.0, d1;

    float mean = calculateMean(data, size);

    for (int i = 0; i < size; i++) {
      // Data 1
      if(isnan(*(data + i))) {
          d1 = pow(mean, 2);
      }
      else {
          d1 = pow(data[i] - mean, 2);
      }
      diff += d1;
    }

    return sqrt(diff / size);
}

// Normalize values to be evaluated for the model.
float* normalize_data(const float *data, size_t size, const float mean_training, const float std_training){
    float* norm = (float*)malloc(sizeof(float) * size);

    //float mean = calculateMean(data, size);
    //float std = calculateStd(data, size);

    for (size_t i = 0; i < size; i++) {
      if(isnan(data[i])) {
          norm[i] = data[i];
      }
      else {
          //norm[i] = (data[i] - mean)/std;
          norm[i] = (data[i] - mean_training)/std_training;
      }
    }

    return norm;  
}