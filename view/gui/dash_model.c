/**
 * dash_model.c — Dashboard Model 实现
 *
 * 通过 ViewInterface 操作，不知道 service 层的具体类型。
 */

#include "view/gui/dash_model.h"
#include <string.h>

void dash_model_init(DashModel *m, ViewInterface *iface, void *ctx) {
    memset(m, 0, sizeof(DashModel));
    m->iface = *iface;
    m->ctx = ctx;
    m->selected_task = -1;
    m->is_training = 0;
}

void dash_model_select(DashModel *m, int task_index) {
    m->selected_task = task_index;
}

void dash_model_start(DashModel *m) {
    if (m->is_training) {
        dash_model_stop(m);
    }
    if (m->selected_task < 0) return;

    m->iface.session_start(m->ctx, m->selected_task);
    m->loss_count = 0;
    m->is_training = 1;
}

int dash_model_step(DashModel *m) {
    if (!m->is_training) return -1;

    int rc = m->iface.session_step(m->ctx);

    m->iface.get_status(m->ctx, &m->status);
    m->loss_count = m->iface.get_loss_history(m->ctx, m->loss_buf, DASH_MAX_LOSS);

    if (rc != 0) {
        m->is_training = 0;
    }
    return rc;
}

void dash_model_stop(DashModel *m) {
    if (m->is_training) {
        m->iface.session_stop(m->ctx);
        m->is_training = 0;
    }
}
