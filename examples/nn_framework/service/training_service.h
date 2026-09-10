/**
 * training_service.h — 训练服务
 *
 * 服务层。编排能力（训练 + 报告），对外暴露 ViewInterface。
 * view 层通过 ViewInterface 获取数据，不直接依赖此头文件。
 *
 * 此头文件仅供 main.c（组装层）使用。
 */

#ifndef SERVICE_TRAINING_H
#define SERVICE_TRAINING_H

#include "capability/data_source.h"
#include "capability/trainer.h"
#include "capability/reporter.h"
#include "service/view_interface.h"

typedef struct {
    Task       **tasks;
    int          task_count;
} TrainingService;

void          training_service_init(TrainingService *svc);
void          training_service_register(TrainingService *svc, Task *task);
TrainResult   training_service_run(TrainingService *svc, int task_index,
                                   TrainCallback callback, void *user_data);

/* ====== ViewInterface 构建 ====== */

/**
 * 从 TrainingService 构建 ViewInterface。
 * main.c 调用此函数，把结果注入到 view 层。
 *
 * @param svc       服务实例
 * @param iface     输出：接口函数表
 * @param ctx       输出：不透明上下文指针
 */
void training_service_get_interface(TrainingService *svc,
                                    ViewInterface *iface,
                                    void **ctx);
void training_service_release_interface(void *ctx);

#endif /* SERVICE_TRAINING_H */
