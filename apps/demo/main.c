/**
 * main.c — 应用入口
 *
 * 只做三件事：初始化 → 注册任务 → 启动菜单。
 * 不包含任何训练逻辑、网络 API 调用。
 */

#include "apps/demo/menu.h"
#include "trainer/reporter.h"
#include "platform/nn_platform.h"
#include <stdlib.h>
#include <time.h>

/* ====== 任务声明 ====== */
extern Task task_sin;
extern Task task_xor;
extern Task task_projectile;
extern Task task_comfort;
extern Task task_sevenseg;
extern Task task_temp_comp;
extern Task task_battery;
extern Task task_gesture;
extern Task task_voice;

int main(void) {
    platform_console_init();
    srand((unsigned int)time(NULL));

    /* 注册所有任务 */
    static Task *all_tasks[] = {
        &task_sin, &task_xor, &task_projectile,
        &task_comfort, &task_sevenseg,
        &task_temp_comp, &task_battery,
        &task_gesture, &task_voice,
    };
    int count = sizeof(all_tasks) / sizeof(all_tasks[0]);

    /* 启动菜单 */
    task_menu_run(all_tasks, count, nn_reporter_progress, NULL);

    return 0;
}
