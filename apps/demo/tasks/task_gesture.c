/**
 * task_gesture.c — 手势/动作识别
 *
 * 输入：12 个特征（IMU 三轴统计特征，归一化）    输出：4 类 one-hot
 *   0=静止, 1=挥手, 2=敲击, 3=摇晃
 */

#include "task.h"
#include <stdlib.h>
#include <math.h>

#define SAMPLE_COUNT 50
#define GRAVITY 9.81
#define NOISE_STD 0.5

static double gauss_rand(double mean, double std) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-10) u1 = 1e-10;
    return mean + sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2) * std;
}

static void extract_features(const double *ax, const double *ay, const double *az, int n, double *f) {
    for (int axis = 0; axis < 3; axis++) {
        const double *d = (axis == 0) ? ax : (axis == 1) ? ay : az;
        double sum = 0, sum2 = 0, peak = 0;
        for (int i = 0; i < n; i++) {
            sum += d[i]; sum2 += d[i] * d[i];
            double a = fabs(d[i]); if (a > peak) peak = a;
        }
        f[axis * 3 + 0] = sum / n;
        f[axis * 3 + 1] = sum2 / n - (sum / n) * (sum / n);
        f[axis * 3 + 2] = peak;
    }
    double energy = 0, zcr = 0, mag_min = 1e10, mag_max = -1e10;
    for (int i = 0; i < n; i++) {
        double mag = sqrt(ax[i]*ax[i] + ay[i]*ay[i] + az[i]*az[i]);
        energy += mag * mag;
        if (mag < mag_min) mag_min = mag;
        if (mag > mag_max) mag_max = mag;
        if (i > 0) {
            double prev = sqrt(ax[i-1]*ax[i-1] + ay[i-1]*ay[i-1] + az[i-1]*az[i-1]);
            if ((mag - prev) * (mag - prev) > 0.01) zcr++;
        }
    }
    f[9]  = energy / n;
    f[10] = zcr / (n - 1);
    f[11] = mag_max - mag_min;
}

static void gen_gesture(int gesture, double *ax, double *ay, double *az) {
    double phase = ((double)rand() / RAND_MAX) * 2.0 * M_PI;
    double amp   = 5.0 + ((double)rand() / RAND_MAX) * 10.0;
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        double t = i * 0.01;
        double nx = gauss_rand(0, NOISE_STD), ny = gauss_rand(0, NOISE_STD), nz = gauss_rand(0, NOISE_STD);
        switch (gesture) {
            case 0: ax[i]=nx; ay[i]=GRAVITY+ny; az[i]=nz; break;
            case 1: ax[i]=amp*sin(6.0*3.14159265358979323846*t+phase)+nx; ay[i]=GRAVITY+ny*0.5; az[i]=amp*0.3*cos(6.0*3.14159265358979323846*t+phase)+nz; break;
            case 2: { double p=amp*1.5*sin(16.0*3.14159265358979323846*t+phase)*exp(-3.0*(t-0.25)); ax[i]=nx*0.5; ay[i]=GRAVITY+ny*0.5; az[i]=p+nz; break; }
            case 3: ax[i]=amp*sin(10.0*3.14159265358979323846*t+phase)+amp*0.5*sin(14.6*3.14159265358979323846*t+phase*1.7)+nx; ay[i]=GRAVITY+amp*0.4*cos(8.2*3.14159265358979323846*t+phase*0.8)+ny; az[i]=amp*0.7*sin(13.4*3.14159265358979323846*t+phase*1.3)+nz; break;
        }
    }
}

static void generate(double *input, double *target) {
    int gesture = rand() % 4;
    double ax[SAMPLE_COUNT], ay[SAMPLE_COUNT], az[SAMPLE_COUNT];
    gen_gesture(gesture, ax, ay, az);
    double raw[12];
    extract_features(ax, ay, az, SAMPLE_COUNT, raw);
    double max[12] = {15,100,25, 15,100,25, 15,100,25, 500,1,40};
    for (int i = 0; i < 12; i++) {
        input[i] = fabs(raw[i]) / max[i];
        if (input[i] > 1) input[i] = 1;
    }
    for (int i = 0; i < 4; i++)
        target[i] = (i == gesture) ? 1.0 : 0.0;
}

Task task_gesture = {
    .name        = "手势动作识别",
    .description = "IMU时序特征 + 分类",
    .input_dim   = 12,
    .output_dim  = 4,
    .generate    = generate,
    .config      = {
        .hidden_count      = 2,
        .hidden_sizes     = {32, 32},
        .hidden_act       = ACT_RELU,
        .output_act       = ACT_SOFTMAX,
        .learning_rate    = 0.005,
        .epochs           = 5000,
        .samples_per_epoch = 100,
    },
};
