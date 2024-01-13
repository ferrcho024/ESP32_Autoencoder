#ifndef MD_UTILS_H
#define MD_UTILS_H

float calc_mse(const float *x, const float *x_hat, const int len);
float calculateMean(const float *data, size_t size);
float calculateStd(const float *data, size_t size);
float* normalize_data(const float *data, size_t size, const float mean_training, const float std_training);


#endif //MD_UTILS_H
