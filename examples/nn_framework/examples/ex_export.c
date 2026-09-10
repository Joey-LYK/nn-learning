/**
 * ex_export.c — 模型导出示例
 *
 * 演示：训练 → 导出 → 导入 → 验证。
 */

#include "component/nn/nn.h"
#include "capability/trainer.h"
#include "capability/data_source.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern Task task_sin;

int main(void) {
    printf("=== Model export example ===\n\n");

    TrainResult result = nn_trainer_run(&task_sin, NULL, NULL);
    printf("Training done: loss=%.6f, time=%.2fs\n\n", result.final_loss, result.elapsed_sec);

    NeuralNet *nn = nn_trainer_build(&task_sin);
    nn_forward(nn, NULL, NULL);

    int size = 0;
    nn_export(nn, NULL, &size);
    printf("Export size: %d bytes\n", size);

    unsigned char *buffer = (unsigned char*)malloc(size);
    nn_export(nn, buffer, &size);
    nn_destroy(nn);

    NeuralNet *nn2 = nn_import(buffer, size);
    if (nn2) {
        printf("Import OK: %d layers, %d params\n", nn2->layer_count, nn2->total_params);

        double input[] = {1.0};
        double output[1];
        nn_predict(nn2, input, output);
        printf("Verify: sin(1.0) -> predict %.6f (true 0.841)\n", output[0]);
        nn_destroy(nn2);
    } else {
        printf("Import failed\n");
    }

    free(buffer);
    printf("\n=== Done ===\n");
    return 0;
}
