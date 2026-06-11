/**
 * ex_minimal.c — 最小示例
 *
 * 演示：不经过 engine/app 层，直接使用 core API。
 * 可独立编译，不依赖任何其他层。
 *
 * 编译：
 *   gcc -Wall -O2 ex_minimal.c ../core/nn.c -I../core -o ex_minimal -lm
 */

#include "core/nn.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("=== 最小示例：直接使用 core API ===\n\n");

    /* 创建网络 */
    NeuralNet *nn = nn_create(0.01);
    nn_add_layer(nn, 2, ACT_LINEAR);   /* 输入层：2 个输入 */
    nn_add_layer(nn, 16, ACT_RELU);     /* 隐藏层：16 个神经元 */
    nn_add_layer(nn, 16, ACT_RELU);     /* 隐藏层：16 个神经元 */
    nn_add_layer(nn, 1, ACT_LINEAR);   /* 输出层：1 个输出 */

    nn_print_structure(nn);

    /* 前向传播 */
    double input[] = {0.5, 0.3};
    double output[1];
    nn_predict(nn, input, output);
    printf("输入 [0.5, 0.3] → 输出 [%.6f]\n\n", output[0]);

    /* 序列化演示 */
    int size = 0;
    nn_export(nn, NULL, &size);
    printf("序列化大小: %d bytes\n", size);

    unsigned char *buffer = (unsigned char*)malloc(size);
    nn_export(nn, buffer, &size);

    NeuralNet *nn2 = nn_import(buffer, size);
    if (nn2) {
        printf("导入成功! params=%d\n", nn2->total_params);
        nn_destroy(nn2);
    } else {
        printf("导入失败\n");
    }

    free(buffer);
    nn_destroy(nn);
    printf("=== 完成 ===\n");
    return 0;
}
