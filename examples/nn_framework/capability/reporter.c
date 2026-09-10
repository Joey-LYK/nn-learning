/**
 * reporter.c — 报告能力实现
 */

#include "capability/reporter.h"
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
    (void)result;
}
