/**
 * view_interface.h — View 层抽象接口
 *
 * view 层与 service/capability/component 完全解耦的唯一桥梁。
 * 只有基本类型和函数指针，不 include 任何其他头文件。
 *
 * service 层负责实现此接口，main.c 负责注入到 view。
 */

#ifndef VIEW_INTERFACE_H
#define VIEW_INTERFACE_H

typedef struct {
    int    current_epoch;
    int    total_epochs;
    double current_loss;
    double elapsed_sec;
    int    is_done;
} ViewTrainingStatus;

typedef struct {
    int         (*get_task_count)(void *ctx);
    const char* (*get_task_name)(void *ctx, int index);
    const char* (*get_task_desc)(void *ctx, int index);

    void        (*session_start)(void *ctx, int task_index);
    int         (*session_step)(void *ctx);
    void        (*session_stop)(void *ctx);

    void        (*get_status)(void *ctx, ViewTrainingStatus *out);
    int         (*get_loss_history)(void *ctx, double *buf, int buf_size);

    void        (*get_train_header)(void *ctx, char *buf, int buf_size);
    void        (*get_validation)(void *ctx, char *buf, int buf_size);
} ViewInterface;

#endif /* VIEW_INTERFACE_H */
