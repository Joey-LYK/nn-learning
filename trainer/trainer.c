/**
 * nn_trainer.c — 训练引擎实现
 */

#include "trainer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* ====== 从 Task config 构建网络 ====== */

NeuralNet* nn_trainer_build(const Task *task) {
    const TaskConfig *cfg = &task->config;
    NeuralNet *nn = nn_create(cfg->learning_rate);

    /* 输入层 */
    nn_add_layer(nn, task->input_dim, ACT_LINEAR);

    /* 隐藏层 */
    for (int i = 0; i < cfg->hidden_count; i++) {
        nn_add_layer(nn, cfg->hidden_sizes[i], (Activation)cfg->hidden_act);
    }

    /* 输出层 */
    nn_add_layer(nn, task->output_dim, (Activation)cfg->output_act);

    return nn;
}

/* ====== 执行完整训练 ====== */

TrainResult nn_trainer_run(const Task *task, TrainCallback callback, void *user_data) {
    const TaskConfig *cfg = &task->config;
    TrainResult result = {0};

    NeuralNet *nn = nn_trainer_build(task);
    nn_print_structure(nn);

    printf("开始训练: %s\n", task->name);
    printf("训练参数: epochs=%d, samples/epoch=%d, lr=%.4f\n\n",
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

    /* ── 验证 ── */
    int is_softmax = (cfg->output_act == ACT_SOFTMAX);
    int test_count = NN_DEFAULT_TEST_SAMPLES;
    double *output = (double*)malloc(sizeof(double) * task->output_dim);
    int correct = 0;

    if (is_softmax) {
        printf("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("训练完成！用时 %.2f 秒\n", result.elapsed_sec);
        printf("── 分类验证（%d 个随机样本）──\n", test_count);
        printf("%-25s %-12s %-12s\n", "输入", "预测类别", "真实类别");
        printf("%-25s %-12s %-12s\n", "───────", "────────", "────────");
    } else {
        printf("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("训练完成！用时 %.2f 秒\n", result.elapsed_sec);
        printf("── 回归验证（%d 个随机样本）──\n", test_count);
        printf("%-25s %-20s %-20s\n", "输入", "预测", "真实值");
        printf("%-25s %-20s %-20s\n", "───────", "────", "──────");
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
            printf("  →  类别 %-2d    真实: 类别 %-2d", pred_class, true_class);
        } else {
            printf("  →  [");
            for (int j = 0; j < task->output_dim; j++)
                printf("%.4f%s", output[j], j < task->output_dim - 1 ? ", " : "]");
            printf("  真实: [");
            for (int j = 0; j < task->output_dim; j++)
                printf("%.4f%s", target[j], j < task->output_dim - 1 ? ", " : "]");
        }
        printf("\n");
    }

    if (is_softmax) {
        printf("\n准确率: %d / %d (%.1f%%)\n", correct, test_count,
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
