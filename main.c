/**
 * main.c — 程序总入口
 *
 * 只做三件事：初始化平台 → 注册 task → 启动视图。
 * 不包含任何业务逻辑。
 */

#include "view/cli/menu.h"
#include "capability/reporter.h"
#include "service/training_service.h"
#include "platform/nn_platform.h"
#include <stdlib.h>
#include <time.h>

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

    TrainingService svc;
    training_service_init(&svc);

    training_service_register(&svc, &task_sin);
    training_service_register(&svc, &task_xor);
    training_service_register(&svc, &task_projectile);
    training_service_register(&svc, &task_comfort);
    training_service_register(&svc, &task_sevenseg);
    training_service_register(&svc, &task_temp_comp);
    training_service_register(&svc, &task_battery);
    training_service_register(&svc, &task_gesture);
    training_service_register(&svc, &task_voice);

    cli_menu_run(&svc, nn_reporter_progress, NULL);

    free(svc.tasks);
    return 0;
}
