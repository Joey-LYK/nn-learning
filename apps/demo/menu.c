/**
 * task_menu.c — 菜单渲染与用户交互实现
 */

#include "apps/demo/menu.h"
#include "trainer/reporter.h"
#include <stdio.h>
#include <stdlib.h>

int task_menu_run(Task **tasks, int count, TrainCallback callback, void *user_data) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║   神经网络训练平台 — 多任务训练演示               ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");

    for (int i = 0; i < count; i++) {
        printf("║  %d. %-44s ║\n", i + 1, tasks[i]->name);
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

    nn_trainer_run(tasks[choice - 1], callback, user_data);
    return choice;
}
