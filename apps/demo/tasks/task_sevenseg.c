/**
 * task_sevenseg.c — 数码管字符识别
 *
 * 输入：7 个值 (七段管 a~g，0/1)    输出：10 类 one-hot (0~9)
 */

#include "task.h"
#include <stdlib.h>

static const int TRUTH_TABLE[10][7] = {
    {1,1,1,1,1,1,0}, /* 0 */
    {0,1,1,0,0,0,0}, /* 1 */
    {1,1,0,1,1,0,1}, /* 2 */
    {1,1,1,1,0,0,1}, /* 3 */
    {0,1,1,0,0,1,1}, /* 4 */
    {1,0,1,1,0,1,1}, /* 5 */
    {1,0,1,1,1,1,1}, /* 6 */
    {1,1,1,0,0,0,0}, /* 7 */
    {1,1,1,1,1,1,1}, /* 8 */
    {1,1,1,1,0,1,1}, /* 9 */
};

static void generate(double *input, double *target) {
    int digit = rand() % 10;
    for (int i = 0; i < 7; i++) {
        /* 10% 概率翻转（模拟噪声） */
        input[i] = (rand() % 10 < 9) ? (double)TRUTH_TABLE[digit][i]
                                      : (double)(1 - TRUTH_TABLE[digit][i]);
    }
    for (int i = 0; i < 10; i++)
        target[i] = (i == digit) ? 1.0 : 0.0;
}

Task task_sevenseg = {
    .name        = "数码管字符识别",
    .description = "传感器噪声容忍",
    .input_dim   = 7,
    .output_dim  = 10,
    .generate    = generate,
    .config      = {
        .hidden_count      = 2,
        .hidden_sizes     = {32, 16},
        .hidden_act       = ACT_RELU,
        .output_act       = ACT_SOFTMAX,
        .learning_rate    = 0.01,
        .epochs           = 3000,
        .samples_per_epoch = 30,
    },
};
