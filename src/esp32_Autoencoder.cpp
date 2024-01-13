/**
 * Use TensorFlow Lite model on real accelerometer data to detect anomalies
 * 
 * NOTE: You will need to install the TensorFlow Lite library:
 * https://www.tensorflow.org/lite/microcontrollers
 * 
 * Author: Shawn Hymel
 * Date: May 6, 2020
 * 
 * License: Beerware
 */

// Library includes
#include <Arduino.h>
#include <math.h>

// Local includes
#include "modelo_df.h"

// Import TensorFlow stuff
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

// We need our utils functions for calculating MAD
extern "C" {
#include "utils.h"
};

// Set to 1 to output debug info to Serial, 0 otherwise
#define DEBUG 1

// Pins
//constexpr int BUZZER_PIN = A1;

// Settings
constexpr float THRESHOLD = 0.3500242427984803;    // Any MSE over this is an anomaly
constexpr float MEAN_TRAINING = 26.403898673843077;    // Mean of the training process
constexpr float STD_TRAINING = 10.86128076630132;    // Standard Desviation of the training process
constexpr int WAIT_TIME = 1000;       // ms between sample sets

// TFLite globals, used for compatibility with Arduino-style sketches
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  // Create an area of memory to use for input, output, and other TensorFlow
  // arrays. You'll need to adjust this by combiling, running, and looking
  // for errors.
  constexpr int kTensorArenaSize = 6 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace
 
/*******************************************************************************
 * Main
 */
 
void setup() {

  // Initialize Serial port for debugging
  #if DEBUG
    Serial.begin(115200);
    while (!Serial);
  #endif

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(modelo_df);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while(1);
  }


  // With all Ops Resolver
  //static tflite::AllOpsResolver micro_mutable_op_resolver;

  // Pull in only needed operations (should match NN layers)
  // Available ops:
  //  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h
  // Based on https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/guide/model_analyzer.ipynb#scrollTo=_jkg6UNtdz8c
  static tflite::MicroMutableOpResolver<7> micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddConv2D();
  micro_mutable_op_resolver.AddTransposeConv();
  micro_mutable_op_resolver.AddStridedSlice();
  micro_mutable_op_resolver.AddShape();
  micro_mutable_op_resolver.AddPack();
  micro_mutable_op_resolver.AddDequantize();
  micro_mutable_op_resolver.AddQuantize();
  
  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
    model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }

  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
 
}

void loop() {

  TfLiteStatus invoke_status;

  float read_data[] = {120.46522};/*, 44.46522, 42.39872, 44.46522, 42.39872, 45.49846, 42.39872, 80.33223, 36.19923, NAN, NAN, 35.16599, 35.16599, 35.16599,
                      34.13274, 37.23248, 39.29898, 40.33223, 40.33223, 39.29898, NAN, 40.33223, 40.33223, 40.33223, 36.19923, 36.19923, 40.33223, 41.36547, 44.46522,
                      46.53171, 46.53171, 47.56496, 42.39872, 41.36547, 45.49846, 51.69795, 50.6647, 48.59821, 45.49846, 43.43197, 41.36547, 41.36547, 43.43197, 47.56496,
                      43.43197, 40.33223, 44.46522, 45.49846, 43.43197, 41.36547, 41.36547, 44.46522, 47.56496, 45.49846, 41.36547, 45.49846, 47.56496, 45.49846, 44.46522};
  */
  size_t size = sizeof(read_data) / sizeof(read_data[0]);

  float* input_data = normalize_data(read_data, size, MEAN_TRAINING, STD_TRAINING);

  // Copiar los datos al tensor de entrada del modelo
  for (int i = 0; i < size; i++) {
      model_input->data.f[i] = input_data[i];
  }

  /*
  Serial.println("\nValores ingresados al modelo");
  for (int pos = 0; pos < size; pos++) {
    Serial.println(model_input->data.f[pos]);
  }
  */

  // Run inference
  invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input");
  }

  // Read predicted y value from output buffer (tensor)
  float pred_vals[size];
  float acum = 0;
  Serial.println();

  //Serial.println("\nValores output después de ejecutado el modelo 1");
  // Reshaping the array for compatibility with 1D model
  for (int pos = 0; pos < size*16; pos+=4) {
    //Serial.println(model_output->data.f[pos]);

    acum += model_output->data.f[pos];

    if (((pos+4) % 16) == 0){
      pred_vals[(int) (pos/16)] = acum/4;
      acum = 0;  
    }

  }

  /*
  Serial.println("\nValores output después de ejecutado el modelo 2");
  for (int pos = 0; pos < size; pos++) {
    Serial.println(pred_vals[pos]);
  }
  */

  #if DEBUG
    Serial.println("Inference result: ");
    for (int pos = 0; pos < size; pos++) {
      float mae_loss = fabs(pred_vals[pos] - input_data[pos]);
      if (mae_loss > THRESHOLD){
        Serial.println("****** OUTLIER ******");
        Serial.print("INPUT DATA: ");
        Serial.println(read_data[pos]);
        Serial.print("MAE: ");
        Serial.println(mae_loss);
        Serial.println();
      }
    
    }
  #endif

  // Liberar la memoria asignada por normalize_data
  free(input_data);
  printf("************ Free Memory: %u bytes ************\n", esp_get_free_heap_size());
}
