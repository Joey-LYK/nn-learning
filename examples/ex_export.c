/**
 * ex_export.c — 模型导出示例
 *
 * 演示：训练一个网络 → 导出为二进制 → 导入并验证。
 *
 * 编译：
 *   gcc -Wall -O2 ex_export.c ../core/nn.c ../trainer/trainer.c
 *       ../trainer/reporter.c ../apps/demo/tasks/task_sin.c -I../core -I../trainer -I../apps/demo/tasks -I../apps/demo
 *       -o ex_export -lm
 */

#include "core/nn.h"
#include "trainer/trainer.h"
#include "apps/demo/tasks/task.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern Task task_sin;

int main(void) {
    printf("=== 模型导出示例 ===\n\n");

    /* 训练一个简单网络 */
    TrainResult result = nn_trainer_run(&task_sin, NULL, NULL);
    printf("训练完成: loss=%.6f, 耗时=%.2fs\n\n", result.final_loss, result.elapsed_sec);

    /* 导出 */
    NeuralNet *nn = nn_trainer_build(&task_sin);
    nn_forward(nn, NULL, NULL);  /* 触发一次让内部缓存有效 */

    int size = 0;
    nn_export(nn, NULL, &size);
    printf("导出大小: %d bytes\n", size);

    unsigned char *buffer = (unsigned char*)malloc(size);
    nn_export(nn, buffer, &size);
    nn_destroy(nn);

    /* 导入 */
    NeuralNet *nn2 = nn_import(buffer, size);
    if (nn2) {
        printf("导入成功: %d 层, %d 参数\n", nn2->layer_count, nn2->total_params);

        /* 验证 */
        double input[] = {1.0};
        double output[1];
        nn_predict(nn2, input, output);
        printf("验证: sin(1.0) → 预测 %.6f (真实 0.841)\n", output[0]);
        nn_destroy(nn2);
    } else {
        printf("导入失败\n");
    }

    free(buffer);
    printf("\n=== 完成 ===\n");
    return 0;
}
