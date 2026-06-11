/**
 * task_temp_comp.c — 传感器温度补偿
 *
 * 输入：3 个值 (ADC读数, 环境温度, 供电电压，归一化)    输出：1 个值 (真实温度)
 */

#include "task.h"
#include <stdlib.h>
#include <math.h>

static double gauss_rand(double mean, double std) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-10) u1 = 1e-10;
    return mean + sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2) * std;
}

static void generate(double *input, double *target) {
    double real_temp = ((double)rand() / RAND_MAX) * 80.0 - 10.0;   /* -10~70°C */
    double env_temp = ((double)rand() / RAND_MAX) * 50.0 + 10.0;    /* 10~60°C */
    double v_supply = ((double)rand() / RAND_MAX) * 0.5 + 4.8;       /* 4.8~5.3V */

    /* NTC 热敏电阻非线性 */
    double beta = 3950.0;
    double T0 = 25.0, R0 = 10000.0;
    double Rt = R0 * exp(beta * (1.0 / (real_temp + 273.15) - 1.0 / (T0 + 273.15)));

    /* 分压 + 自发热 */
    double R_series = 10000.0;
    double self_heat = 0.0;
    double I_sense = v_supply / (R_series + Rt);
    self_heat = I_sense * I_sense * 50.0 * 0.001; /* ~50Ω 串联电阻 */

    /* ADC 读数 */
    double v_ntc = v_supply * Rt / (R_series + Rt);
    double v_adc = v_ntc + gauss_rand(0, 0.02) + self_heat * 0.01;

    /* 电压波动 */
    v_supply += gauss_rand(0, 0.05);

    /* 归一化 */
    input[0] = (v_adc - 0.0) / 5.0;
    input[1] = (env_temp - 10.0) / 60.0;
    input[2] = (v_supply - 4.8) / 0.5;

    if (input[0] < 0) input[0] = 0;
    if (input[0] > 1) input[0] = 1;
    if (input[1] < 0) input[1] = 0;
    if (input[1] > 1) input[1] = 1;
    if (input[2] < 0) input[2] = 0;
    if (input[2] > 1) input[2] = 1;

    target[0] = (real_temp + 10.0) / 80.0;  /* 归一化到 [0, 1] */
}

Task task_temp_comp = {
    .name        = "传感器温度补偿",
    .description = "回归 + 非线性校正",
    .input_dim   = 3,
    .output_dim  = 1,
    .generate    = generate,
    .config      = {
        .hidden_count      = 2,
        .hidden_sizes     = {32, 16},
        .hidden_act       = ACT_RELU,
        .output_act       = ACT_LINEAR,
        .learning_rate    = 0.005,
        .epochs           = 5000,
        .samples_per_epoch = 100,
    },
};
