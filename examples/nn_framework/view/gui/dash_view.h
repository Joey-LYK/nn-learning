/**
 * dash_view.h — Dashboard View (MVC)
 *
 * PainterEngine UI 控件创建 + 刷新。
 * 只依赖 dash_model.h（不直接依赖 service/capability/component）。
 */

#ifndef DASH_VIEW_H
#define DASH_VIEW_H

#include "view/gui/dash_model.h"
#include <PainterEngine.h>

#define MAX_TASKS_VIEW 16

typedef struct {
    PX_Object *buttons[MAX_TASKS_VIEW];
    PX_Object *oscilloscope;
    PX_Object *processbar;
    PX_Object *label_status;
    PX_Object *label_epoch;
    PX_Object *label_loss;
    PX_Object *timer;
    int        button_count;
    int        osc_allocated;
} DashView;

void dash_view_init(DashView *v, DashModel *m);
void dash_view_update(DashView *v, DashModel *m);

#endif /* DASH_VIEW_H */
