/**
 * task_battery.c — 电池 SOC 估计
 *
 * 输入：3 个值 (端电压, 电流, 温度，归一化)    输出：1 个值 (SOC 0~1)
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

static double ocv_from_soc(double soc) {
    /* 三段式 OCV-SOC 曲线 */
    if (soc < 0.1) return 3.0 + (4.2 - 3.0) * soc / 0.1;
    if (soc < 0.9) return 4.2 + (3.8 - 4.2) * (soc - 0.1) / 0.8;
    return 3.8 + (3.0 - 3.8) * (soc - 0.9) / 0.1;
}

static double internal_resistance(double soc, double temp_c) {
    double T_K = temp_c + 273.15;
    double Ea = 30000.0;  /* 活化能 */
    double R0 = 0.05;     /* SOC=1 时内阻 */
    return R0 * exp((Ea / 8.314) * (1.0 / T_K - 1.0 / 298.15)) / soc;
}

static void generate(double *input, double *target) {
    double soc      = ((double)rand() / RAND_MAX);                  /* 0~1 */
    double temp    = ((double)rand() / RAND_MAX) * 45.0 + (-10.0);   /* -10~35°C */
    double current = ((double)rand() / RAND_MAX) * 2.0 - 1.0;      /* -1~1A */
    double r       = internal_resistance(soc, temp);

    double V_full  = 4.2;
    double V_empty = 3.0;
    double v_terminal = ocv_from_soc(soc) - current * r;

    /* 测量噪声 */
    v_terminal += gauss_rand(0, 0.01);
    current += gauss_rand(0, 0.05);

    input[0] = (v_terminal - (V_empty - 0.2)) / (V_full + 0.2 - (V_empty - 0.2));
    input[1] = (current + 1.0) / 2.0;
    input[2] = (temp - (-10.0)) / 45.0;

    if (input[0] < 0) input[0] = 0;
    if (input[0] > 1) input[0] = 1;
    if (input[1] < 0) input[1] = 0;
    if (input[1] > 1) input[1] = 1;
    if (input[2] < 0) input[2] = 0;
    if (input[2] > 1) input[2] = 1;

    target[0] = soc;
}

Task task_battery = {
    .name        = "电池SOC估计",
    .description = "嵌入式BMS核心功能",
    .input_dim   = 3,
    .output_dim  = 1,
    .generate    = generate,
    .config      = {
        .hidden_count      = 3,
        .hidden_sizes     = {64, 32, 16},
        .hidden_act       = ACT_RELU,
        .output_act       = ACT_LINEAR,
        .learning_rate    = 0.005,
        .epochs           = 5000,
        .samples_per_epoch = 100,
    },
};
