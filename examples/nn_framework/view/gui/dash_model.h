/**
 * dash_model.h — Dashboard Model (MVC)
 *
 * 仪表盘状态数据。只依赖 view_interface.h，不知道 service 的具体类型。
 * view/controller 通过此 model 获取所有训练数据。
 */

#ifndef DASH_MODEL_H
#define DASH_MODEL_H

#include "service/view_interface.h"

#define DASH_MAX_LOSS 8192

typedef struct {
    ViewInterface      iface;
    void              *ctx;
    int                selected_task;
    int                is_training;
    ViewTrainingStatus status;
    double             loss_buf[DASH_MAX_LOSS];
    int                loss_count;
} DashModel;

void dash_model_init(DashModel *m, ViewInterface *iface, void *ctx);
void dash_model_select(DashModel *m, int task_index);
void dash_model_start(DashModel *m);
int  dash_model_step(DashModel *m);
void dash_model_stop(DashModel *m);

#endif /* DASH_MODEL_H */
