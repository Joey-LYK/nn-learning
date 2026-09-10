/**
 * nn_config.h — 框架编译时配置
 *
 * 集中管理全局编译开关、默认参数、view 层启用。
 * 修改此文件后需重新编译整个工程。
 */

#ifndef NN_CONFIG_H
#define NN_CONFIG_H

/* ====== 数据类型 ====== */
#if !defined(NN_USE_FLOAT) || defined(NN_USE_DOUBLE)
    typedef double nn_float_t;
    #define NN_FLOAT_FMT  "%.6f"
    #define NN_MATH_FN(x) x
#else
    typedef float  nn_float_t;
    #define NN_FLOAT_FMT  "%.4f"
    #define NN_MATH_FN(x) x##f
#endif

/* ====== 激活函数类型 ====== */
typedef enum {
    ACT_SIGMOID,
    ACT_TANH,
    ACT_RELU,
    ACT_LINEAR,
    ACT_SOFTMAX,
} Activation;

/* ====== 框架限制 ====== */
#define NN_MAX_HIDDEN_LAYERS     4

/* ====== 输出/报告配置 ====== */
#define NN_OUTPUT_UNICODE_BAR    1
#define NN_DEFAULT_TEST_SAMPLES  10
#define NN_REPORT_INTERVAL_DIV   10

/* ====== 调试开关 ====== */
#define NN_DEBUG_PRINT_WEIGHTS   0
#define NN_DEBUG_PRINT_GRADIENT  0

/* ====== View 层开关 ====== */
#ifndef NN_VIEW_CLI
#define NN_VIEW_CLI    1
#endif

#ifndef NN_VIEW_GUI
#define NN_VIEW_GUI    1
#endif

#endif /* NN_CONFIG_H */
