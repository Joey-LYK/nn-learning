/**
 * dash_ctrl.h — Dashboard Controller (MVC)
 *
 * 事件处理 + 训练启动控制。
 */

#ifndef DASH_CTRL_H
#define DASH_CTRL_H

#include "view/gui/dash_model.h"
#include "view/gui/dash_view.h"

typedef struct {
    DashModel *model;
    DashView  *view;
} DashCtrl;

void dash_ctrl_init(DashCtrl *c, DashModel *m, DashView *v);

#endif /* DASH_CTRL_H */
