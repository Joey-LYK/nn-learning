/**
 * trainer.c — 训练能力实现
 */

#include "capability/trainer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

NeuralNet* nn_trainer_build(const Task *task) {
    const TaskConfig *cfg = &task->config;
    NeuralNet *nn = nn_create(cfg->learning_rate);

    nn_add_layer(nn, task->input_dim, ACT_LINEAR);

    for (int i = 0; i < cfg->hidden_count; i++) {
        nn_add_layer(nn, cfg->hidden_sizes[i], (Activation)cfg->hidden_act);
    }

    nn_add_layer(nn, task->output_dim, (Activation)cfg->output_act);

    return nn;
}

TrainResult nn_trainer_run(const Task *task, TrainCallback callback, void *user_data) {
    const TaskConfig *cfg = &task->config;
    TrainResult result = {0};

    NeuralNet *nn = nn_trainer_build(task);
    nn_print_structure(nn);

    printf("start training: %s\n", task->name);
    printf("epochs=%d, samples/epoch=%d, lr=%.4f\n\n",
           cfg->epochs, cfg->samples_per_epoch, cfg->learning_rate);

    double *input  = (double*)malloc(sizeof(double) * task->input_dim);
    double *target = (double*)malloc(sizeof(double) * task->output_dim);

    int report_interval = cfg->epochs / NN_REPORT_INTERVAL_DIV;
    if (report_interval < 1) report_interval = 1;

    clock_t start = clock();

    for (int ep = 0; ep < cfg->epochs; ep++) {
        double total_loss = 0.0;

        for (int s = 0; s < cfg->samples_per_epoch; s++) {
            task->generate(input, target);
            total_loss += nn_train_step(nn, input, target);
        }

        double avg_loss = total_loss / cfg->samples_per_epoch;
        result.final_loss = avg_loss;

        if (callback && ((ep + 1) % report_interval == 0 || ep == 0)) {
            TrainReport report;
            report.epoch = ep + 1;
            report.total_epochs = cfg->epochs;
            report.avg_loss = avg_loss;
            report.elapsed_sec = (double)(clock() - start) / CLOCKS_PER_SEC;
            callback(&report, user_data);
        }
    }

    clock_t end = clock();
    result.elapsed_sec = (double)(end - start) / CLOCKS_PER_SEC;

    int is_softmax = (cfg->output_act == ACT_SOFTMAX);
    int test_count = NN_DEFAULT_TEST_SAMPLES;
    double *output = (double*)malloc(sizeof(double) * task->output_dim);
    int correct = 0;

    if (is_softmax) {
        printf("\nclassification validation (%d samples)\n", test_count);
    } else {
        printf("\nregression validation (%d samples)\n", test_count);
    }

    for (int i = 0; i < test_count; i++) {
        task->generate(input, target);
        nn_predict(nn, input, output);

        printf("[");
        for (int j = 0; j < task->input_dim; j++)
            printf("%.3f%s", input[j], j < task->input_dim - 1 ? ", " : "]");

        if (is_softmax) {
            int pred_class = 0, true_class = 0;
            for (int j = 1; j < task->output_dim; j++) {
                if (output[j] > output[pred_class])  pred_class  = j;
                if (target[j] > target[true_class])  true_class = j;
            }
            if (pred_class == true_class) correct++;
            printf("  ->  pred %d  true %d", pred_class, true_class);
        } else {
            printf("  ->  [");
            for (int j = 0; j < task->output_dim; j++)
                printf("%.4f%s", output[j], j < task->output_dim - 1 ? ", " : "]");
            printf("  true: [");
            for (int j = 0; j < task->output_dim; j++)
                printf("%.4f%s", target[j], j < task->output_dim - 1 ? ", " : "]");
        }
        printf("\n");
    }

    if (is_softmax) {
        printf("\naccuracy: %d / %d (%.1f%%)\n", correct, test_count,
               100.0 * correct / test_count);
        result.correct_count = correct;
    } else {
        result.correct_count = -1;
    }
    result.test_count = test_count;

    printf("\n");

    free(input);
    free(target);
    free(output);
    nn_destroy(nn);

    return result;
}

double nn_trainer_train_epoch(NeuralNet *nn, const Task *task) {
    double *input  = (double*)malloc(sizeof(double) * task->input_dim);
    double *target = (double*)malloc(sizeof(double) * task->output_dim);
    double total_loss = 0.0;

    for (int s = 0; s < task->config.samples_per_epoch; s++) {
        task->generate(input, target);
        total_loss += nn_train_step(nn, input, target);
    }

    free(input);
    free(target);
    return total_loss / task->config.samples_per_epoch;
}
