/**
 * ex_minimal.c — 最小示例
 *
 * 演示：直接使用 component/nn API，不依赖其他层。
 */

#include "component/nn/nn.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("=== Minimal example: component/nn API ===\n\n");

    NeuralNet *nn = nn_create(0.01);
    nn_add_layer(nn, 2, ACT_LINEAR);
    nn_add_layer(nn, 16, ACT_RELU);
    nn_add_layer(nn, 16, ACT_RELU);
    nn_add_layer(nn, 1, ACT_LINEAR);

    nn_print_structure(nn);

    double input[] = {0.5, 0.3};
    double output[1];
    nn_predict(nn, input, output);
    printf("Input [0.5, 0.3] -> Output [%.6f]\n\n", output[0]);

    int size = 0;
    nn_export(nn, NULL, &size);
    printf("Serialize size: %d bytes\n", size);

    unsigned char *buffer = (unsigned char*)malloc(size);
    nn_export(nn, buffer, &size);

    NeuralNet *nn2 = nn_import(buffer, size);
    if (nn2) {
        printf("Import OK! params=%d\n", nn2->total_params);
        nn_destroy(nn2);
    } else {
        printf("Import failed\n");
    }

    free(buffer);
    nn_destroy(nn);
    printf("=== Done ===\n");
    return 0;
}
