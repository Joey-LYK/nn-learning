/**
 * training_service.c — 训练服务实现
 *
 * ServiceContext 包装 svc + session，对 view 层只暴露 void* ctx。
 * 所有 vi 函数统一操作 ServiceContext。
 */

#include "service/training_service.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* ====== TrainingService 基础操作 ====== */

void training_service_init(TrainingService *svc) {
    svc->tasks = NULL;
    svc->task_count = 0;
}

void training_service_register(TrainingService *svc, Task *task) {
    int old = svc->task_count;
    svc->tasks = (Task**)realloc(svc->tasks, sizeof(Task*) * (old + 1));
    svc->tasks[old] = task;
    svc->task_count = old + 1;
}

TrainResult training_service_run(TrainingService *svc, int task_index,
                                 TrainCallback callback, void *user_data) {
    if (task_index < 0 || task_index >= svc->task_count) {
        TrainResult r = {0};
        return r;
    }
    return nn_trainer_run(svc->tasks[task_index], callback, user_data);
}

/* ====== 内部结构（view 层不可见） ====== */

#define MAX_LOSS_POINTS 8192

typedef enum {
    SESSION_IDLE,
    SESSION_TRAINING,
    SESSION_DONE
} SessionState;

typedef struct {
    SessionState    state;
    NeuralNet      *nn;
    const Task     *task;
    int             current_epoch;
    int             total_epochs;
    double          loss_history[MAX_LOSS_POINTS];
    int             loss_count;
    double          elapsed_sec;
    double          start_time;
} TrainingSession;

typedef struct {
    TrainingService *svc;
    TrainingSession *session;
} ServiceContext;

/* ====== ViewInterface 函数实现 ====== */

static int vi_get_task_count(void *ctx) {
    ServiceContext *sc = (ServiceContext*)ctx;
    return sc->svc->task_count;
}

static const char* vi_get_task_name(void *ctx, int index) {
    ServiceContext *sc = (ServiceContext*)ctx;
    if (index < 0 || index >= sc->svc->task_count) return "";
    return sc->svc->tasks[index]->name;
}

static const char* vi_get_task_desc(void *ctx, int index) {
    ServiceContext *sc = (ServiceContext*)ctx;
    if (index < 0 || index >= sc->svc->task_count) return "";
    return sc->svc->tasks[index]->description;
}

static void vi_session_start(void *ctx, int task_index) {
    ServiceContext *sc = (ServiceContext*)ctx;

    if (sc->session) {
        nn_destroy(sc->session->nn);
        free(sc->session);
        sc->session = NULL;
    }

    if (task_index < 0 || task_index >= sc->svc->task_count) return;

    TrainingSession *s = (TrainingSession*)calloc(1, sizeof(TrainingSession));
    s->task = sc->svc->tasks[task_index];
    s->nn = nn_trainer_build(s->task);
    s->total_epochs = s->task->config.epochs;
    s->start_time = (double)clock() / CLOCKS_PER_SEC;
    s->state = SESSION_TRAINING;
    sc->session = s;
}

static int vi_session_step(void *ctx) {
    ServiceContext *sc = (ServiceContext*)ctx;
    TrainingSession *session = sc->session;
    if (!session || session->state != SESSION_TRAINING) return -1;
    if (session->loss_count >= MAX_LOSS_POINTS) {
        session->state = SESSION_DONE;
        return 1;
    }

    double *input  = (double*)malloc(sizeof(double) * session->task->input_dim);
    double *target = (double*)malloc(sizeof(double) * session->task->output_dim);
    double total_loss = 0.0;

    for (int s = 0; s < session->task->config.samples_per_epoch; s++) {
        session->task->generate(input, target);
        total_loss += nn_train_step(session->nn, input, target);
    }

    double loss = total_loss / session->task->config.samples_per_epoch;
    session->loss_history[session->loss_count] = loss;
    session->loss_count++;
    session->current_epoch++;
    session->elapsed_sec = (double)clock() / CLOCKS_PER_SEC - session->start_time;

    if (session->current_epoch >= session->total_epochs) {
        session->state = SESSION_DONE;
    }

    free(input);
    free(target);
    return (session->state == SESSION_DONE) ? 1 : 0;
}

static void vi_session_stop(void *ctx) {
    ServiceContext *sc = (ServiceContext*)ctx;
    if (!sc->session) return;
    nn_destroy(sc->session->nn);
    free(sc->session);
    sc->session = NULL;
}

static void vi_get_status(void *ctx, ViewTrainingStatus *out) {
    ServiceContext *sc = (ServiceContext*)ctx;
    if (!sc->session || !out) {
        if (out) memset(out, 0, sizeof(ViewTrainingStatus));
        return;
    }
    TrainingSession *session = sc->session;
    out->current_epoch = session->current_epoch;
    out->total_epochs  = session->total_epochs;
    out->current_loss  = (session->loss_count > 0)
                         ? session->loss_history[session->loss_count - 1] : 0;
    out->elapsed_sec   = session->elapsed_sec;
    out->is_done       = (session->state == SESSION_DONE);
}

static int vi_get_loss_history(void *ctx, double *buf, int buf_size) {
    ServiceContext *sc = (ServiceContext*)ctx;
    if (!sc->session || !buf) return 0;
    TrainingSession *session = sc->session;
    int count = (session->loss_count < buf_size) ? session->loss_count : buf_size;
    memcpy(buf, session->loss_history, sizeof(double) * count);
    return count;
}

static void vi_get_train_header(void *ctx, char *buf, int buf_size) {
    ServiceContext *sc = (ServiceContext*)ctx;
    if (!sc->session || !buf) { if (buf) buf[0] = '\0'; return; }
    TrainingSession *s = sc->session;
    const Task *task = s->task;
    NeuralNet *nn = s->nn;

    static const char *act_names[] = {
        "Sigmoid", "Tanh", "ReLU", "Linear", "Softmax"
    };

    int p = 0;
    p += snprintf(buf + p, buf_size - p, "\n");
    p += snprintf(buf + p, buf_size - p,
        "+==========================================+\n");
    p += snprintf(buf + p, buf_size - p,
        "|        Neural Network Structure          |\n");
    p += snprintf(buf + p, buf_size - p,
        "+------------------------------------------+\n");
    p += snprintf(buf + p, buf_size - p,
        "|  Total layers:    %-3d                   |\n", nn->layer_count);
    p += snprintf(buf + p, buf_size - p,
        "|  Total params:    %-6d                |\n", nn->total_params);
    p += snprintf(buf + p, buf_size - p,
        "|  Learning rate:   %-8.5f              |\n", nn->learning_rate);
    p += snprintf(buf + p, buf_size - p,
        "+------------------------------------------+\n");

    for (int L = 0; L < nn->layer_count; L++) {
        NNLayer *layer = &nn->layers[L];
        const char *role = (L == 0) ? "Input " :
                           (L == nn->layer_count - 1) ? "Output" : "Hidden";
        p += snprintf(buf + p, buf_size - p,
            "|  L%d %s: %3d neurons  [%s]    |\n",
            L, role, layer->size, act_names[layer->activation]);
        if (L > 0) {
            int w = layer->size * layer->input_size;
            p += snprintf(buf + p, buf_size - p,
                "|       weights: %d, biases: %d        |\n", w, layer->size);
        }
    }
    p += snprintf(buf + p, buf_size - p,
        "+==========================================+\n\n");

    p += snprintf(buf + p, buf_size - p,
        "start training: %s\n", task->name);
    p += snprintf(buf + p, buf_size - p,
        "epochs=%d, samples/epoch=%d, lr=%.4f\n\n",
        task->config.epochs, task->config.samples_per_epoch, task->config.learning_rate);
}

static void vi_get_validation(void *ctx, char *buf, int buf_size) {
    ServiceContext *sc = (ServiceContext*)ctx;
    if (!sc->session || !buf) { if (buf) buf[0] = '\0'; return; }
    TrainingSession *s = sc->session;
    const Task *task = s->task;

    int is_softmax = (task->config.output_act == ACT_SOFTMAX);
    int test_count = NN_DEFAULT_TEST_SAMPLES;

    double *input  = (double*)malloc(sizeof(double) * task->input_dim);
    double *target = (double*)malloc(sizeof(double) * task->output_dim);
    double *output = (double*)malloc(sizeof(double) * task->output_dim);

    int p = 0;
    if (is_softmax) {
        p += snprintf(buf + p, buf_size - p,
            "\nclassification validation (%d samples)\n", test_count);
    } else {
        p += snprintf(buf + p, buf_size - p,
            "\nregression validation (%d samples)\n", test_count);
    }

    int correct = 0;
    for (int i = 0; i < test_count; i++) {
        task->generate(input, target);
        nn_predict(s->nn, input, output);

        p += snprintf(buf + p, buf_size - p, "[");
        for (int j = 0; j < task->input_dim; j++)
            p += snprintf(buf + p, buf_size - p, "%.3f%s",
                input[j], j < task->input_dim - 1 ? ", " : "]");

        if (is_softmax) {
            int pred_class = 0, true_class = 0;
            for (int j = 1; j < task->output_dim; j++) {
                if (output[j] > output[pred_class])  pred_class  = j;
                if (target[j] > target[true_class])  true_class = j;
            }
            if (pred_class == true_class) correct++;
            p += snprintf(buf + p, buf_size - p, "  ->  pred %d  true %d\n",
                pred_class, true_class);
        } else {
            p += snprintf(buf + p, buf_size - p, "  ->  [");
            for (int j = 0; j < task->output_dim; j++)
                p += snprintf(buf + p, buf_size - p, "%.4f%s",
                    output[j], j < task->output_dim - 1 ? ", " : "]");
            p += snprintf(buf + p, buf_size - p, "  true: [");
            for (int j = 0; j < task->output_dim; j++)
            p += snprintf(buf + p, buf_size - p, "%.4f%s",
                target[j], j < task->output_dim - 1 ? ", " : "]");
            p += snprintf(buf + p, buf_size - p, "\n");
        }
    }

    if (is_softmax) {
        p += snprintf(buf + p, buf_size - p, "\naccuracy: %d / %d (%.1f%%)\n",
            correct, test_count, 100.0 * correct / test_count);
    }

    free(input);
    free(target);
    free(output);
}

/* ====== 构建 / 释放 ViewInterface ====== */

void training_service_get_interface(TrainingService *svc,
                                    ViewInterface *iface,
                                    void **ctx) {
    ServiceContext *sc = (ServiceContext*)calloc(1, sizeof(ServiceContext));
    sc->svc = svc;
    sc->session = NULL;

    iface->get_task_count  = vi_get_task_count;
    iface->get_task_name   = vi_get_task_name;
    iface->get_task_desc   = vi_get_task_desc;
    iface->session_start   = vi_session_start;
    iface->session_step    = vi_session_step;
    iface->session_stop    = vi_session_stop;
    iface->get_status      = vi_get_status;
    iface->get_loss_history = vi_get_loss_history;
    iface->get_train_header = vi_get_train_header;
    iface->get_validation   = vi_get_validation;
    *ctx = sc;
}

void training_service_release_interface(void *ctx) {
    ServiceContext *sc = (ServiceContext*)ctx;
    if (!sc) return;
    if (sc->session) {
        nn_destroy(sc->session->nn);
        free(sc->session);
    }
    free(sc);
}
