/**
 * task_xor.c — XOR 逻辑门
 *
 * 输入：2 个值 (0/1)    输出：1 个值 (0/1)
 */

#include "capability/data_source.h"
#include <stdlib.h>

static void generate(double *input, double *target) {
    int a = rand() % 2;
    int b = rand() % 2;
    input[0]  = (double)a;
    input[1]  = (double)b;
    target[0] = (double)(a ^ b);
}

Task task_xor = {
    .name        = "XOR 逻辑门",
    .description = "理解非线性的必要性",
    .input_dim   = 2,
    .output_dim  = 1,
    .generate    = generate,
    .config      = {
        .hidden_count      = 1,
        .hidden_sizes     = {4},
        .hidden_act       = ACT_SIGMOID,
        .output_act       = ACT_SIGMOID,
        .learning_rate    = 0.1,
        .epochs           = 2000,
        .samples_per_epoch = 20,
    },
};
