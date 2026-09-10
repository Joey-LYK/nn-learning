/**
 * dash_view.c — Dashboard View 实现
 *
 * 所有训练数据通过 DashModel（内部通过 ViewInterface）获取。
 * 不直接访问 service/capability 的任何类型。
 */

#include "view/gui/dash_view.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static DashModel *g_model;
static DashView  *g_view;

static px_color task_colors[MAX_TASKS_VIEW];

static void init_colors(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    task_colors[0] = PX_COLOR(255, 100, 200, 100);
    task_colors[1] = PX_COLOR(255, 100, 150, 255);
    task_colors[2] = PX_COLOR(255, 255, 150, 50);
    task_colors[3] = PX_COLOR(255, 50, 200, 200);
    task_colors[4] = PX_COLOR(255, 200, 100, 200);
    task_colors[5] = PX_COLOR(255, 150, 255, 100);
    task_colors[6] = PX_COLOR(255, 255, 200, 100);
    task_colors[7] = PX_COLOR(255, 100, 100, 200);
    task_colors[8] = PX_COLOR(255, 200, 200, 100);
}

static void on_button_click(PX_Object *obj, PX_Object_Event e, void *ptr) {
    (void)obj; (void)e;
    int index = (int)(intptr_t)ptr;
    dash_model_stop(g_model);
    dash_model_select(g_model, index);
    dash_model_start(g_model);

    if (g_view->timer) {
        PX_Object_TimerResume(g_view->timer);
    }
    g_view->osc_allocated = 0;
}

static void on_timer(PX_Object *obj, PX_Object_Event e, void *ptr) {
    (void)obj; (void)e; (void)ptr;
    if (!g_model->is_training) return;

    dash_model_step(g_model);
    dash_view_update(g_view, g_model);

    if (g_model->status.is_done) {
        PX_Object_TimerPause(g_view->timer);
    }
}

void dash_view_init(DashView *v, DashModel *m) {
    g_model = m;
    g_view = v;
    memset(v, 0, sizeof(DashView));
    init_colors();

    int count = m->iface.get_task_count(m->ctx);
    int btn_w = 160, btn_h = 32;
    int btn_x = 20, btn_y_start = 60;

    for (int i = 0; i < count && i < MAX_TASKS_VIEW; i++) {
        const char *name = m->iface.get_task_name(m->ctx, i);
        v->buttons[i] = PX_Object_PushButtonCreate(
            mp, root, btn_x, btn_y_start + i * (btn_h + 8),
            btn_w, btn_h, name, NULL
        );
        PX_ObjectRegisterEvent(v->buttons[i], PX_OBJECT_EVENT_EXECUTE,
                               on_button_click, (void*)(intptr_t)i);
        v->button_count++;
    }

    int osc_x = 210, osc_y = 20, osc_w = 780, osc_h = 400;
    v->oscilloscope = PX_Object_OscilloscopeCreate(
        mp, root, osc_x, osc_y, osc_w, osc_h, NULL
    );
    PX_Object_OscilloscopeSetHorizontalMin(v->oscilloscope, 0);
    PX_Object_OscilloscopeSetLeftVerticalMin(v->oscilloscope, 0);
    PX_Object_OscilloscopeSetBorderColor(v->oscilloscope, PX_COLOR(255, 100, 100, 100));
    PX_Object_OscilloscopeSetGuidesVisible(v->oscilloscope, PX_TRUE);
    PX_Object_OscilloscopeSetScaleEnabled(v->oscilloscope, PX_TRUE);

    v->processbar = PX_Object_ProcessBarCreate(mp, root, 210, 460, 780, 24);
    PX_Object_ProcessBarSetMax(v->processbar, 100);
    PX_Object_ProcessBarSetValue(v->processbar, 0);
    PX_Object_ProcessBarSetColor(v->processbar, PX_COLOR(255, 0, 180, 120));

    v->label_status = PX_Object_LabelCreate(
        mp, root, 210, 500, 400, 28, "Ready", NULL,
        PX_COLOR(255, 220, 220, 220)
    );
    v->label_epoch = PX_Object_LabelCreate(
        mp, root, 210, 530, 400, 28, "", NULL,
        PX_COLOR(255, 200, 200, 200)
    );
    v->label_loss = PX_Object_LabelCreate(
        mp, root, 620, 530, 400, 28, "", NULL,
        PX_COLOR(255, 200, 200, 200)
    );

    v->timer = PX_Object_TimerCreate(mp, root, 0, 16, 0);
    PX_ObjectRegisterEvent(v->timer, PX_OBJECT_EVENT_TIMEOUT, on_timer, NULL);
    PX_Object_TimerPause(v->timer);

    v->osc_allocated = 0;
}

void dash_view_update(DashView *v, DashModel *m) {
    if (!m->is_training && !m->status.is_done) return;

    ViewTrainingStatus *s = &m->status;

    if (!v->osc_allocated && m->loss_count > 0) {
        PX_Object_OscilloscopeClearData(v->oscilloscope);

        PX_Object_OscilloscopeData data;
        memset(&data, 0, sizeof(data));
        data.MapHorizontalArray = (px_double*)malloc(sizeof(px_double) * s->total_epochs);
        data.MapVerticalArray   = (px_double*)malloc(sizeof(px_double) * s->total_epochs);
        data.Normalization = 1.0;
        data.Color = task_colors[m->selected_task % MAX_TASKS_VIEW];
        data.linewidth = 2;
        data.Map = PX_OBJECT_OSCILLOSCOPEDATA_MAP_LEFT;
        data.ID = 0;
        data.Visibled = 1;
        data.Size = 0;
        PX_Object_OscilloscopeAddData(v->oscilloscope, data);

        PX_Object_OscilloscopeSetHorizontalMax(v->oscilloscope, (px_double)s->total_epochs);
        PX_Object_OscilloscopeSetLeftVerticalMax(v->oscilloscope, m->loss_buf[0] * 1.2);
        v->osc_allocated = 1;
    }

    if (v->osc_allocated) {
        PX_Object_OscilloscopeData *d = PX_Object_OscilloscopeGetOscilloscopeData(v->oscilloscope, 0);
        if (d && m->loss_count > 0) {
            int idx = m->loss_count - 1;
            d->MapHorizontalArray[idx] = (px_double)idx;
            d->MapVerticalArray[idx] = m->loss_buf[idx];
            d->Size = m->loss_count;

            double max_loss = 0;
            for (int i = 0; i < m->loss_count; i++) {
                if (m->loss_buf[i] > max_loss) max_loss = m->loss_buf[i];
            }
            PX_Object_OscilloscopeSetLeftVerticalMax(v->oscilloscope, max_loss * 1.2);
        }
    }

    int pct = (s->total_epochs > 0) ? s->current_epoch * 100 / s->total_epochs : 0;
    PX_Object_ProcessBarSetValue(v->processbar, pct);

    char buf[128];
    const char *task_name = m->iface.get_task_name(m->ctx, m->selected_task);
    if (s->is_done) {
        snprintf(buf, sizeof(buf), "DONE - %s", task_name);
    } else {
        snprintf(buf, sizeof(buf), "Training - %s", task_name);
    }
    PX_Object_LabelSetText(v->label_status, buf);

    snprintf(buf, sizeof(buf), "Epoch: %d/%d  Time: %.1fs",
             s->current_epoch, s->total_epochs, s->elapsed_sec);
    PX_Object_LabelSetText(v->label_epoch, buf);

    if (m->loss_count > 0) {
        snprintf(buf, sizeof(buf), "Loss: %.6f", m->loss_buf[m->loss_count - 1]);
        PX_Object_LabelSetText(v->label_loss, buf);
    }
}
