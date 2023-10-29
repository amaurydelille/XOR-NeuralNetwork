#include "xor.h"
#include <math.h>
#include <stdio.h>

int main() {
    NeuralNetwork nn;
    initialize_network(&nn);

    double inputs[][INPUT_NEURONS] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 2}, {8, 1}, {9, 0}, {1, 1}};
    double targets[] = {0, 1, 1, 0, 0, 1, 1, 0};
    double accuracy = 0;

    train(&nn, inputs, targets, 8);

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < INPUT_NEURONS; j++) {
            nn.input[j] = inputs[i][j];
        }
        forward_propagation(&nn);
        printf("Input: %f, %f - Output: %f\n", inputs[i][0], inputs[i][1], nn.output[0]);
        if (fabs(nn.output[0] - targets[i]) < 0.5) 
            accuracy++;
    }

    printf("XOR ACCURACY : %f\n", accuracy/8);
}
