/**
 * menu.c — CLI 视图实现
 *
 * 通过 ViewInterface 获取数据，不知道 service 的具体类型。
 * 内置进度条显示，不依赖 reporter。
 */

#include "view/cli/menu.h"
#include "nn_config.h"
#include <stdio.h>
#include <stdlib.h>

static void cli_print_bar(int current, int total) {
    int bar_w = 40;
    int filled = (total > 0) ? current * bar_w / total : 0;
    printf("  [");
    for (int i = 0; i < bar_w; i++) {
        if (i < filled) printf("#");
        else            printf("-");
    }
    printf("] %3d%%", (total > 0) ? current * 100 / total : 0);
}

void cli_menu_run(ViewInterface *iface, void *ctx) {
    int count = iface->get_task_count(ctx);

    printf("\n");
    printf("+--------------------------------------------------+\n");
    printf("|   Neural Network Training Platform               |\n");
    printf("+--------------------------------------------------+\n");

    for (int i = 0; i < count; i++) {
        printf("|  %d. %-45s |\n", i + 1, iface->get_task_name(ctx, i));
    }

    printf("| -------------------------------------------------|\n");
    printf("|  0. Exit                                         |\n");
    printf("+--------------------------------------------------+\n");
    printf("\nSelect task [0-%d]: ", count);

    int choice;
    if (scanf("%d", &choice) != 1) return;

    if (choice == 0) {
        printf("Bye!\n");
        return;
    }
    if (choice < 1 || choice > count) {
        printf("Invalid: %d\n", choice);
        return;
    }

    int task_index = choice - 1;

    iface->session_start(ctx, task_index);

    char header[2048];
    iface->get_train_header(ctx, header, sizeof(header));
    printf("%s", header);

    ViewTrainingStatus status;
    iface->get_status(ctx, &status);
    int report_interval = status.total_epochs / 10;
    if (report_interval < 1) report_interval = 1;

    while (1) {
        int rc = iface->session_step(ctx);
        iface->get_status(ctx, &status);

        if (status.current_epoch % report_interval == 0 || rc != 0) {
            printf("\r  Epoch %d/%d  Loss: %.6f  Time: %.1fs ",
                   status.current_epoch, status.total_epochs,
                   status.current_loss, status.elapsed_sec);
            cli_print_bar(status.current_epoch, status.total_epochs);
            fflush(stdout);
        }

        if (rc != 0) break;
    }

    printf("\n\nTraining complete! Final loss: %.6f  Time: %.1fs\n",
           status.current_loss, status.elapsed_sec);

    char validation[8192];
    iface->get_validation(ctx, validation, sizeof(validation));
    printf("%s", validation);

    printf("\n");
    iface->session_stop(ctx);
}
