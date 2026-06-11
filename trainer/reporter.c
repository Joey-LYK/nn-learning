/**
 * nn_reporter.c — 输出/报告服务实现
 */

#include "reporter.h"
#include <stdio.h>

void nn_reporter_progress(const TrainReport *report, void *user_data) {
    (void)user_data;
    int bar_width = 20;
    int filled = report->epoch * bar_width / report->total_epochs;

    printf("[Epoch %4d/%4d] |", report->epoch, report->total_epochs);

#if NN_OUTPUT_UNICODE_BAR
    for (int b = 0; b < bar_width; b++)
        printf("%s", b < filled ? "█" : "░");
#else
    for (int b = 0; b < bar_width; b++)
        printf("%s", b < filled ? "#" : "-");
#endif

    printf("|  Loss: %10.6f\n", report->avg_loss);
    fflush(stdout);
}

void nn_reporter_summary(const TrainResult *result) {
    /* 训练完成后已在 nn_trainer_run 中打印摘要 */
    (void)result;
}

void nn_reporter_validate_regression(const Task *task, NeuralNet *nn, int count) {
    (void)task; (void)nn; (void)count;
    /* 回归验证集成在 nn_trainer_run 中 */
}

void nn_reporter_validate_classification(const Task *task, NeuralNet *nn, int count) {
    (void)task; (void)nn; (void)count;
    /* 分类验证集成在 nn_trainer_run 中 */
}
