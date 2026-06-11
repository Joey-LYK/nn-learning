/**
 * training_service.c — 训练服务实现
 */

#include "service/training_service.h"
#include <stdlib.h>

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

Task* training_service_get_task(TrainingService *svc, int index) {
    if (index < 0 || index >= svc->task_count) return NULL;
    return svc->tasks[index];
}

int training_service_get_count(TrainingService *svc) {
    return svc->task_count;
}
