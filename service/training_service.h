/**
 * training_service.h — 训练服务
 *
 * 服务层。编排能力（训练 + 报告），对外提供业务接口。
 * view 层只与此文件交互，不直接调用 capability。
 */

#ifndef SERVICE_TRAINING_H
#define SERVICE_TRAINING_H

#include "capability/data_source.h"
#include "capability/trainer.h"
#include "capability/reporter.h"

typedef struct {
    Task       **tasks;
    int          task_count;
} TrainingService;

void          training_service_init(TrainingService *svc);
void          training_service_register(TrainingService *svc, Task *task);
TrainResult   training_service_run(TrainingService *svc, int task_index,
                                   TrainCallback callback, void *user_data);
Task*         training_service_get_task(TrainingService *svc, int index);
int           training_service_get_count(TrainingService *svc);

#endif /* SERVICE_TRAINING_H */
