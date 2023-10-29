#include "xor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double reLU(double x) {
    if (x > 0)
        return x;
    else
        return 0;
}

double xavier_weight(int n_in, int n_out) {
    double variance = 2.0 / (n_in + n_out);
    double standard_deviation = sqrt(variance);
    return ((double)rand() / RAND_MAX) * 2 * standard_deviation - standard_deviation;
}


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

double binary_cross_entropy_loss(double y, double y_pred) {
    double epsilon = 1e-15;
    y_pred = fmax(epsilon, fmin(1 - epsilon, y_pred));
    return -(y*log(y_pred) + (1-y)*log(1-y_pred));
}


double random_weight() {
    return ((double)rand() / RAND_MAX) * 2 - 1;
}

void initialize_network(NeuralNetwork *nn) {
    for(size_t i = 0; i < INPUT_NEURONS; i++)
    {
        nn->bias_h[i] = random_weight();

        for(size_t j = 0; j < HIDDEN_NEURONS; j++)
            nn->weights_ih[i][j] = xavier_weight(INPUT_NEURONS, HIDDEN_NEURONS);
        
    }

    for(size_t i = 0; i < OUTPUT_NEURONS; i++)
    {
        nn->bias_o[i] = random_weight();
        for(size_t j = 0; j < HIDDEN_NEURONS; j++)
            nn->weights_ho[i][j] = xavier_weight(HIDDEN_NEURONS, OUTPUT_NEURONS);        
    }
}

void forward_propagation(NeuralNetwork *nn) {
    for(size_t i = 0; i < HIDDEN_NEURONS; i++)
    {
        nn->hidden[i] = 0;
        for(size_t j = 0; j < INPUT_NEURONS; j++)
        {
            nn->hidden[i] += nn->input[j] * nn->weights_ih[i][j];
        }
        nn->hidden[i] += nn->bias_h[i];
        nn->hidden[i] = sigmoid(nn->hidden[i]);
    }

    for(size_t i = 0; i < OUTPUT_NEURONS; i++)
    {
        nn->output[i] = 0;
        for(size_t j = 0; j < HIDDEN_NEURONS; j++)
        {
            nn->output[i] += nn->hidden[j] * nn->weights_ho[i][j];
        }
        nn->output[i] += nn->bias_o[i];
        nn->output[i] = sigmoid(nn->output[i]);
    }
}

void backpropagation(NeuralNetwork *nn, double target) {
    double output_errors[OUTPUT_NEURONS];
    double output_deltas[OUTPUT_NEURONS];

    for(size_t i = 0; i < OUTPUT_NEURONS; i++)
    {
        output_errors[i] = target - nn->output[i];
        output_deltas[i] = output_errors[i] * sigmoid_derivative(nn->output[i]);
        nn->bias_o[i] += output_deltas[i] * LEARNING_RATE;

        for(size_t j = 0; j < HIDDEN_NEURONS; j++)
            nn->weights_ho[i][j] += nn->hidden[j] * output_deltas[i] * LEARNING_RATE;
    }

    double hidden_errors[HIDDEN_NEURONS];
    double hidden_deltas[HIDDEN_NEURONS];

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
    {
        hidden_errors[i] = 0;
        for (size_t j = 0; j < OUTPUT_NEURONS; j++) 
            hidden_errors[i] += output_deltas[j] * nn->weights_ho[j][i];
        
        hidden_deltas[i] = hidden_errors[i] * sigmoid_derivative(nn->hidden[i]);
    }

    for(size_t i = 0; i < HIDDEN_NEURONS; i++)
    {
        nn->bias_h[i] += hidden_deltas[i] * LEARNING_RATE;
        for(size_t j = 0; j < INPUT_NEURONS; j++)
            nn->weights_ih[i][j] += nn->input[j] * hidden_deltas[i] * LEARNING_RATE;
    }
}


void train(NeuralNetwork *nn, double inputs[][INPUT_NEURONS], double targets[], int samples) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_error = 0;
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < INPUT_NEURONS; j++) {
                nn->input[j] = inputs[i][j];
            }

            forward_propagation(nn);

            double error = 0;
            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                error += binary_cross_entropy_loss(targets[i], nn->output[j]);
            }
            total_error += error;

            backpropagation(nn, targets[i]);
        }

        total_error /= samples;

        if (epoch % 1000 == 0) {
            printf("Epoch %d: Error = %f\n", epoch, total_error);
        }
    }
}


