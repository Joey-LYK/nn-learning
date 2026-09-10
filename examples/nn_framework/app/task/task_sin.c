/**
 * task_sin.c — sin(x) 函数拟合
 *
 * 输入：1 个值 x ∈ [0, 2π]    输出：1 个值 y ≈ sin(x)
 */

#include "capability/data_source.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void generate(double *input, double *target) {
    double x = ((double)rand() / RAND_MAX) * 2.0 * M_PI;
    input[0]  = x;
    target[0] = sin(x);
}

Task task_sin = {
    .name        = "sin(x) 函数拟合",
    .description = "理解基本训练流程",
    .input_dim   = 1,
    .output_dim  = 1,
    .generate    = generate,
    .config      = {
        .hidden_count      = 2,
        .hidden_sizes     = {16, 16},
        .hidden_act       = ACT_RELU,
        .output_act       = ACT_LINEAR,
        .learning_rate    = 0.01,
        .epochs           = 5000,
        .samples_per_epoch = 50,
    },
};
