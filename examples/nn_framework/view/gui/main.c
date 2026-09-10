/**
 * main.c — GUI Dashboard 入口
 *
 * PainterEngine 应用入口。
 * 被 #define main px_main 重命名，真正的 main() 在 PainterEngine 的 px_main.c 中。
 *
 * 作为组装层：创建 service → 获取 ViewInterface → 注入到 view 层。
 */

#include "PainterEngine.h"
#include "view/gui/dash_model.h"
#include "view/gui/dash_view.h"
#include "view/gui/dash_ctrl.h"
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

static TrainingService g_svc;
static ViewInterface   g_iface;
static void           *g_ctx;
static DashModel       g_model;
static DashView        g_view;
static DashCtrl        g_ctrl;

static void register_tasks(void) {
    training_service_register(&g_svc, &task_sin);
    training_service_register(&g_svc, &task_xor);
    training_service_register(&g_svc, &task_projectile);
    training_service_register(&g_svc, &task_comfort);
    training_service_register(&g_svc, &task_sevenseg);
    training_service_register(&g_svc, &task_temp_comp);
    training_service_register(&g_svc, &task_battery);
    training_service_register(&g_svc, &task_gesture);
    training_service_register(&g_svc, &task_voice);
}

int main() {
    platform_console_init();
    srand((unsigned int)time(NULL));

    PainterEngine_Initialize(1024, 600);
    PainterEngine_SetWindowText("NN Training Dashboard");

    training_service_init(&g_svc);
    register_tasks();

    training_service_get_interface(&g_svc, &g_iface, &g_ctx);

    dash_model_init(&g_model, &g_iface, g_ctx);
    dash_view_init(&g_view, &g_model);
    dash_ctrl_init(&g_ctrl, &g_model, &g_view);

    return 0;
}
