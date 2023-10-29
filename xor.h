#pragma once

#define INPUT_NEURONS 2
#define HIDDEN_NEURONS 2
#define OUTPUT_NEURONS 1
#define LEARNING_RATE 0.1
#define EPOCHS 500000
#define RANDINT 1

typedef struct {
    double input[INPUT_NEURONS];
    double hidden[HIDDEN_NEURONS];
    double output[OUTPUT_NEURONS];
    double weights_ih[HIDDEN_NEURONS][INPUT_NEURONS];
    double weights_ho[OUTPUT_NEURONS][HIDDEN_NEURONS];
    double bias_h[HIDDEN_NEURONS];
    double bias_o[OUTPUT_NEURONS];
} NeuralNetwork;

double reLU(double x);
double xavier_weight(int n_in, int n_out);
double sigmoid(double x);
double sigmoid_derivative(double x);
double binary_cross_entropy_loss(double y, double y_pred);
double random_weight();
void initialize_network(NeuralNetwork* neuralnetwork);
void train(NeuralNetwork *nn, double inputs[][INPUT_NEURONS], double targets[], int samples);
void backpropagation(NeuralNetwork *nn, double target);
void forward_propagation(NeuralNetwork *nn);