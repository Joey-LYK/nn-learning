/**
 * nn.h — MLP 神经网络组件
 *
 * 纯计算组件。提供前向传播、反向传播、权重更新、序列化。
 * 不知道"训练循环"和"任务"的概念。
 */

#ifndef NN_H
#define NN_H

#include "nn_config.h"

typedef struct {
    int input_size;
    int size;

    double *weights;
    double *biases;

    double *net;
    double *out;
    double *error;

    Activation activation;
} NNLayer;

typedef struct {
    NNLayer *layers;
    int layer_count;
    double learning_rate;
    double total_loss;
    int total_params;
} NeuralNet;

NeuralNet* nn_create(double learning_rate);
void nn_add_layer(NeuralNet *nn, int size, Activation activation);
void nn_destroy(NeuralNet *nn);

void nn_forward(NeuralNet *nn, double *input, double *output);
void nn_predict(NeuralNet *nn, double *input, double *output);

double nn_backward(NeuralNet *nn, double *target);
void nn_update(NeuralNet *nn);
double nn_train_step(NeuralNet *nn, double *input, double *target);

void nn_print_structure(NeuralNet *nn);

int nn_export(NeuralNet *nn, void *buffer, int *size);
NeuralNet* nn_import(void *buffer, int size);

#endif /* NN_H */
