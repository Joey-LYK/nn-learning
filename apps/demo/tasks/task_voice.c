/**
 * task_voice.c — 语音命令识别
 *
 * 输入：13 个特征（模拟 MFCC 频谱包络，归一化）    输出：5 类 one-hot
 *   0=启动 1=停止 2=左转 3=右转 4=加速
 */

#include "task.h"
#include <stdlib.h>
#include <math.h>

#define N_FEATURES 13
#define N_CLASSES  5

static double gauss_rand(double mean, double std) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-10) u1 = 1e-10;
    return mean + sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2) * std;
}

static void generate(double *input, double *target) {
    int cmd = rand() % N_CLASSES;
    double freq[N_FEATURES] = {0};
    double f0 = 300.0 + cmd * 200.0; /* 每个命令的基频不同 */
    /* 模拟 MFCC 频谱包络：基频 + 谐波 */
    freq[0] = f0 * 0.5 + gauss_rand(f0 * 0.45, f0 * 0.1);
    for (int i = 1; i < N_FEATURES - 2; i++)
        freq[i] = freq[0] * (1.0 + i * 0.3) + gauss_rand(0, f0 * 0.05);
    freq[N_FEATURES - 2] = f0 * 8.0 + gauss_rand(0, f0 * 0.2); /* 高频能量 */
    freq[N_FEATURES - 1] = gauss_rand(0, 0.3);                 /* 噪声底 */

    double max_val = 0;
    for (int i = 0; i < N_FEATURES; i++)
        if (fabs(freq[i]) > max_val) max_val = fabs(freq[i]);
    for (int i = 0; i < N_FEATURES; i++) {
        input[i] = fabs(freq[i]) / (max_val * 1.2);
        if (input[i] > 1.0) input[i] = 1.0;
    }

    for (int i = 0; i < N_CLASSES; i++)
        target[i] = (i == cmd) ? 1.0 : 0.0;
}

Task task_voice = {
    .name        = "语音命令识别",
    .description = "音频特征 + 多类分类",
    .input_dim   = N_FEATURES,
    .output_dim  = N_CLASSES,
    .generate    = generate,
    .config      = {
        .hidden_count      = 2,
        .hidden_sizes     = {64, 32},
        .hidden_act       = ACT_RELU,
        .output_act       = ACT_SOFTMAX,
        .learning_rate    = 0.005,
        .epochs           = 5000,
        .samples_per_epoch = 200,
    },
};
