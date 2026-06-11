/**
 * dash_ctrl.c — Dashboard Controller 实现
 */

#include "view/gui/dash_ctrl.h"

void dash_ctrl_init(DashCtrl *c, DashModel *m, DashView *v) {
    c->model = m;
    c->view = v;
}
