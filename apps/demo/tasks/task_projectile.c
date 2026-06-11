/**
 * task_projectile.c — 抛体运动落点预测
 *
 * 输入：2 个值 (角度°, 初速度 m/s)    输出：1 个值 (水平距离 m)
 */

#include "task.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef G
#define G 9.81
#endif

static void generate(double *input, double *target) {
    double angle = ((double)rand() / RAND_MAX) * 80.0 + 10.0;  /* 10~90° */
    double v0    = ((double)rand() / RAND_MAX) * 40.0 + 10.0;  /* 10~50 m/s */
    double rad   = angle * M_PI / 180.0;
    input[0] = angle / 90.0;   /* 归一化到 [0,1] */
    input[1] = v0 / 50.0;     /* 归一化到 [0,1] */

    double t_flight = 2.0 * v0 * sin(rad) / G;
    target[0] = (v0 * cos(rad) * t_flight) / 255.0;  /* 归一化到 [0,1] */
}

Task task_projectile = {
    .name        = "抛体运动",
    .description = "理解归一化和大数据量",
    .input_dim   = 2,
    .output_dim  = 1,
    .generate    = generate,
    .config      = {
        .hidden_count      = 2,
        .hidden_sizes     = {64, 128},
        .hidden_act       = ACT_SIGMOID,
        .output_act       = ACT_LINEAR,
        .learning_rate    = 0.005,
        .epochs           = 100,
        .samples_per_epoch = 500,
    },
};
