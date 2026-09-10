/**
 * nn.c — MLP 神经网络组件实现
 */

#include "component/nn/nn.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double nn_gauss_rand(void) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static double activate(double x, Activation type) {
    switch (type) {
        case ACT_SIGMOID: return 1.0 / (1.0 + exp(-x));
        case ACT_TANH:    return tanh(x);
        case ACT_RELU:    return x > 0 ? x : 0;
        case ACT_LINEAR:  return x;
        case ACT_SOFTMAX: return x;
        default:          return x;
    }
}

static double activate_derivative(double out, double net, Activation type) {
    switch (type) {
        case ACT_SIGMOID: return out * (1.0 - out);
        case ACT_TANH:    return 1.0 - out * out;
        case ACT_RELU:    return net > 0 ? 1.0 : 0;
        case ACT_LINEAR:  return 1.0;
        case ACT_SOFTMAX: return 1.0;
        default:          return 1.0;
    }
}

NeuralNet* nn_create(double learning_rate) {
    NeuralNet *nn = (NeuralNet*)malloc(sizeof(NeuralNet));
    if (!nn) return NULL;
    memset(nn, 0, sizeof(NeuralNet));
    nn->learning_rate = learning_rate;
    return nn;
}

void nn_add_layer(NeuralNet *nn, int size, Activation activation) {
    int old_count = nn->layer_count;
    nn->layers = (NNLayer*)realloc(nn->layers, sizeof(NNLayer) * (old_count + 1));
    nn->layer_count = old_count + 1;

    NNLayer *layer = &nn->layers[old_count];
    memset(layer, 0, sizeof(NNLayer));
    layer->size = size;
    layer->activation = activation;

    if (old_count > 0) {
        NNLayer *prev = &nn->layers[old_count - 1];
        layer->input_size = prev->size;

        int w_count = size * prev->size;
        layer->weights = (double*)malloc(sizeof(double) * w_count);
        layer->biases  = (double*)malloc(sizeof(double) * size);
        layer->net     = (double*)malloc(sizeof(double) * size);
        layer->out     = (double*)malloc(sizeof(double) * size);
        layer->error   = (double*)malloc(sizeof(double) * size);

        double scale = sqrt(2.0 / (prev->size + size));
        for (int i = 0; i < w_count; i++)
            layer->weights[i] = nn_gauss_rand() * scale;
        for (int i = 0; i < size; i++)
            layer->biases[i] = 0.0;

        nn->total_params += w_count + size;
    } else {
        layer->net = (double*)malloc(sizeof(double) * size);
        layer->out = (double*)malloc(sizeof(double) * size);
    }
}

void nn_destroy(NeuralNet *nn) {
    if (!nn) return;
    for (int i = 0; i < nn->layer_count; i++) {
        NNLayer *layer = &nn->layers[i];
        free(layer->weights);
        free(layer->biases);
        free(layer->net);
        free(layer->out);
        free(layer->error);
    }
    free(nn->layers);
    free(nn);
}

void nn_forward(NeuralNet *nn, double *input, double *output) {
    NNLayer *layers = nn->layers;
    int n = nn->layer_count;

    for (int i = 0; i < layers[0].size; i++) {
        layers[0].net[i] = input[i];
        layers[0].out[i] = input[i];
    }

    for (int L = 1; L < n; L++) {
        NNLayer *cur  = &layers[L];
        NNLayer *prev = &layers[L - 1];

        for (int i = 0; i < cur->size; i++) {
            double sum = 0.0;
            for (int j = 0; j < prev->size; j++)
                sum += prev->out[j] * cur->weights[i * cur->input_size + j];
            cur->net[i] = sum + cur->biases[i];
            cur->out[i] = activate(cur->net[i], cur->activation);
        }

        if (cur->activation == ACT_SOFTMAX) {
            double max_val = cur->net[0];
            for (int i = 1; i < cur->size; i++)
                if (cur->net[i] > max_val) max_val = cur->net[i];
            double sum_exp = 0.0;
            for (int i = 0; i < cur->size; i++) {
                cur->out[i] = exp(cur->net[i] - max_val);
                sum_exp += cur->out[i];
            }
            for (int i = 0; i < cur->size; i++)
                cur->out[i] /= sum_exp;
        }
    }

    if (output) {
        NNLayer *out_layer = &layers[n - 1];
        for (int i = 0; i < out_layer->size; i++)
            output[i] = out_layer->out[i];
    }
}

double nn_backward(NeuralNet *nn, double *target) {
    NNLayer *layers = nn->layers;
    int n = nn->layer_count;
    NNLayer *out_layer = &layers[n - 1];
    double loss = 0.0;

    if (out_layer->activation == ACT_SOFTMAX) {
        for (int i = 0; i < out_layer->size; i++) {
            out_layer->error[i] = out_layer->out[i] - target[i];
            loss -= target[i] * log(out_layer->out[i] + 1e-10);
        }
    } else {
        for (int i = 0; i < out_layer->size; i++) {
            double diff = out_layer->out[i] - target[i];
            out_layer->error[i] = diff * activate_derivative(out_layer->out[i],
                                                               out_layer->net[i],
                                                               out_layer->activation);
            loss += diff * diff;
        }
    }

    for (int L = n - 2; L >= 1; L--) {
        NNLayer *cur  = &layers[L];
        NNLayer *next = &layers[L + 1];
        for (int j = 0; j < cur->size; j++) {
            double back_sum = 0.0;
            for (int k = 0; k < next->size; k++)
                back_sum += next->error[k] * next->weights[k * cur->size + j];
            cur->error[j] = back_sum * activate_derivative(cur->out[j],
                                                             cur->net[j],
                                                             cur->activation);
        }
    }

    nn->total_loss = loss;
    return loss;
}

void nn_update(NeuralNet *nn) {
    NNLayer *layers = nn->layers;
    int n = nn->layer_count;
    double lr = nn->learning_rate;

    for (int L = 1; L < n; L++) {
        NNLayer *cur  = &layers[L];
        NNLayer *prev = &layers[L - 1];
        for (int i = 0; i < cur->size; i++) {
            double delta = cur->error[i];
            for (int j = 0; j < prev->size; j++) {
                double gradient = delta * prev->out[j];
                cur->weights[i * cur->input_size + j] -= lr * gradient;
            }
            cur->biases[i] -= lr * delta;
        }
    }
}

double nn_train_step(NeuralNet *nn, double *input, double *target) {
    nn_forward(nn, input, NULL);
    double loss = nn_backward(nn, target);
    nn_update(nn);
    return loss;
}

void nn_predict(NeuralNet *nn, double *input, double *output) {
    nn_forward(nn, input, output);
}

void nn_print_structure(NeuralNet *nn) {
    const char *act_names[] = {
        [ACT_SIGMOID] = "Sigmoid",
        [ACT_TANH]    = "Tanh",
        [ACT_RELU]    = "ReLU",
        [ACT_LINEAR]  = "Linear",
        [ACT_SOFTMAX] = "Softmax",
    };

    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║        Neural Network Structure          ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  Total layers:    %-3d                   ║\n", nn->layer_count);
    printf("║  Total params:    %-6d                ║\n", nn->total_params);
    printf("║  Learning rate:   %-8.5f              ║\n", nn->learning_rate);
    printf("╠══════════════════════════════════════════╣\n");

    for (int L = 0; L < nn->layer_count; L++) {
        NNLayer *layer = &nn->layers[L];
        const char *role = (L == 0) ? "Input " :
                           (L == nn->layer_count - 1) ? "Output" : "Hidden";
        printf("║  L%d %s: %3d neurons  [%s]    ║\n",
               L, role, layer->size, act_names[layer->activation]);
        if (L > 0) {
            int w = layer->size * layer->input_size;
            printf("║       weights: %d, biases: %d        ║\n", w, layer->size);
        }
    }
    printf("╚══════════════════════════════════════════╝\n\n");
}

int nn_export(NeuralNet *nn, void *buffer, int *size) {
    int total = 0;
    total += sizeof(int) * 3;
    total += sizeof(double);
    for (int L = 0; L < nn->layer_count; L++) {
        NNLayer *layer = &nn->layers[L];
        total += sizeof(int) * 2;
        if (L > 0) {
            int w_count = layer->size * layer->input_size;
            total += sizeof(double) * w_count;
            total += sizeof(double) * layer->size;
        }
    }
    *size = total;

    if (!buffer) return 0;

    unsigned char *p = (unsigned char*)buffer;
    int magic = 0x4E4E3031;
    int layer_count = nn->layer_count;
    int flags = 0;

    memcpy(p, &magic, sizeof(int));        p += sizeof(int);
    memcpy(p, &layer_count, sizeof(int));  p += sizeof(int);
    memcpy(p, &flags, sizeof(int));        p += sizeof(int);
    memcpy(p, &nn->learning_rate, sizeof(double)); p += sizeof(double);

    for (int L = 0; L < nn->layer_count; L++) {
        NNLayer *layer = &nn->layers[L];
        int sz = layer->size;
        int act  = (int)layer->activation;
        memcpy(p, &sz, sizeof(int)); p += sizeof(int);
        memcpy(p, &act,  sizeof(int)); p += sizeof(int);
        if (L > 0) {
            int w_count = layer->size * layer->input_size;
            memcpy(p, layer->weights, sizeof(double) * w_count);  p += sizeof(double) * w_count;
            memcpy(p, layer->biases,  sizeof(double) * sz);     p += sizeof(double) * sz;
        }
    }
    return 0;
}

NeuralNet* nn_import(void *buffer, int size) {
    (void)size;
    unsigned char *p = (unsigned char*)buffer;
    int magic, layer_count, flags;
    double lr;

    memcpy(&magic, p, sizeof(int)); p += sizeof(int);
    if (magic != 0x4E4E3031) return NULL;

    memcpy(&layer_count, p, sizeof(int)); p += sizeof(int);
    memcpy(&flags, p, sizeof(int));       p += sizeof(int);
    (void)flags;

    memcpy(&lr, p, sizeof(double)); p += sizeof(double);

    NeuralNet *nn = nn_create(lr);
    if (!nn) return NULL;

    for (int L = 0; L < layer_count; L++) {
        int sz, act;
        memcpy(&sz,  p, sizeof(int)); p += sizeof(int);
        memcpy(&act, p, sizeof(int)); p += sizeof(int);
        nn_add_layer(nn, sz, (Activation)act);

        if (L > 0) {
            NNLayer *layer = &nn->layers[L];
            int w_count = sz * layer->input_size;
            memcpy(layer->weights, p, sizeof(double) * w_count); p += sizeof(double) * w_count;
            memcpy(layer->biases,  p, sizeof(double) * sz);      p += sizeof(double) * sz;
        }
    }
    return nn;
}
