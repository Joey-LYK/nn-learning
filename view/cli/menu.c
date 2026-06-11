/**
 * menu.c — CLI 视图实现
 */

#include "view/cli/menu.h"
#include "capability/reporter.h"
#include <stdio.h>
#include <stdlib.h>

int cli_menu_run(TrainingService *svc, TrainCallback callback, void *user_data) {
    int count = training_service_get_count(svc);

    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║   神经网络训练平台 — 多任务训练演示               ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");

    for (int i = 0; i < count; i++) {
        Task *t = training_service_get_task(svc, i);
        printf("║  %d. %-44s ║\n", i + 1, t->name);
    }

    printf("║ ───────────────────────────────────────────────── ║\n");
    printf("║  0. 退出                                       ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("\n请选择任务 [0-%d]: ", count);

    int choice;
    if (scanf("%d", &choice) != 1)
        return -1;

    if (choice == 0) {
        printf("再见！\n");
        return 0;
    }
    if (choice < 1 || choice > count) {
        printf("无效选择: %d\n", choice);
        return -1;
    }

    training_service_run(svc, choice - 1, callback, user_data);
    return choice;
}
