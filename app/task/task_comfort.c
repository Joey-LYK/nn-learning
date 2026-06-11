/**
 * task_comfort.c — 环境舒适度分类
 *
 * 输入：2 个值 (温度, 湿度，归一化)    输出：4 类 one-hot
 *   0=冷, 1=舒适, 2=闷, 3=热
 */

#include "capability/data_source.h"
#include <stdlib.h>
#include <math.h>

#define DATASET_SIZE 160
#define TEMP_MIN 5.0
#define TEMP_MAX 42.0
#define HUMI_MIN 15.0
#define HUMI_MAX 100.0

static double dataset[DATASET_SIZE][2];
static int    labels[DATASET_SIZE];

static void build_dataset(void) {
    static int built = 0;
    if (built) return;
    built = 1;

    int idx = 0;
    for (int t = 5; t <= 40 && idx < DATASET_SIZE; t += 3) {
        for (int h = 15; h <= 100 && idx < DATASET_SIZE; h += 8) {
            dataset[idx][0] = (double)t + ((rand() % 3 - 1) * 0.5);
            dataset[idx][1] = (double)h + ((rand() % 3 - 1) * 1.0);

            double temp = dataset[idx][0], humi = dataset[idx][1];
            int jitter = rand() % 3 - 1;
            int label;
            if (temp > 33 + jitter) label = 3;
            else if (temp < 18 - jitter) label = 0;
            else if (humi <= 65 + jitter * 3 && temp <= 28 + jitter) label = 1;
            else label = 2;
            labels[idx] = label;
            idx++;
        }
    }
    while (idx < DATASET_SIZE) {
        dataset[idx][0] = 5.0 + ((double)rand() / RAND_MAX) * 37.0;
        dataset[idx][1] = 15.0 + ((double)rand() / RAND_MAX) * 85.0;
        double temp = dataset[idx][0], humi = dataset[idx][1];
        if (temp > 33) labels[idx] = 3;
        else if (temp < 18) labels[idx] = 0;
        else if (humi <= 65 && temp <= 28) labels[idx] = 1;
        else labels[idx] = 2;
        idx++;
    }
}

static void generate(double *input, double *target) {
    build_dataset();
    int idx = rand() % DATASET_SIZE;
    input[0] = (dataset[idx][0] - TEMP_MIN) / (TEMP_MAX - TEMP_MIN);
    input[1] = (dataset[idx][1] - HUMI_MIN) / (HUMI_MAX - HUMI_MIN);
    for (int i = 0; i < 4; i++)
        target[i] = (i == labels[idx]) ? 1.0 : 0.0;
}

Task task_comfort = {
    .name        = "环境舒适度分类",
    .description = "真实场景分类入门",
    .input_dim   = 2,
    .output_dim  = 4,
    .generate    = generate,
    .config      = {
        .hidden_count      = 2,
        .hidden_sizes     = {16, 16},
        .hidden_act       = ACT_RELU,
        .output_act       = ACT_SOFTMAX,
        .learning_rate    = 0.01,
        .epochs           = 3000,
        .samples_per_epoch = 50,
    },
};
