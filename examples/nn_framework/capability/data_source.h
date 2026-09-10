/**
 * data_source.h — 数据采集能力
 *
 * 定义数据源接口（Task）。每个 app/task/task_xxx.c 实现一个 Task 实例。
 * 能力层定义接口，app 层提供具体实现。
 */

#ifndef DATA_SOURCE_H
#define DATA_SOURCE_H

#include "nn_config.h"

typedef void (*DataGenerator)(nn_float_t *input, nn_float_t *target);

typedef struct {
    int        hidden_count;
    int        hidden_sizes[NN_MAX_HIDDEN_LAYERS];
    int        hidden_act;
    int        output_act;
    double     learning_rate;
    int        epochs;
    int        samples_per_epoch;
} TaskConfig;

typedef struct {
    const char   *name;
    const char   *description;
    int           input_dim;
    int           output_dim;
    DataGenerator generate;
    TaskConfig    config;
} Task;

#endif /* DATA_SOURCE_H */
