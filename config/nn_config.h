/**
 * nn_config.h — 编译时配置
 *
 * 集中管理框架的全局编译开关和默认参数。
 * 修改此文件后需重新编译整个工程。
 */

#ifndef NN_CONFIG_H
#define NN_CONFIG_H

/* ====== 激活函数类型 ====== */

typedef enum {
    ACT_SIGMOID,   /* S 形曲线，输出 (0,1) */
    ACT_TANH,      /* S 形曲线，输出 (-1,1) */
    ACT_RELU,      /* max(0,x)，计算快，深层网络首选 */
    ACT_LINEAR,    /* f(x)=x，回归问题输出层 */
    ACT_SOFTMAX,   /* 多分类输出层，输出和为 1 */
} Activation;

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

/* ====== 框架限制 ====== */
#define NN_MAX_HIDDEN_LAYERS     4    /* 框架支持的最大隐藏层数 */

/* ====== 输出/报告配置 ====== */
#define NN_OUTPUT_UNICODE_BAR    1    /* 1: Unicode 进度条(█░)  0: ASCII 进度条(#+-) */
#define NN_DEFAULT_TEST_SAMPLES  10   /* 验证阶段默认测试样本数 */
#define NN_REPORT_INTERVAL_DIV   10   /* 进度报告间隔 = epochs / 此值，最小为 1 */

/* ====== 调试开关 ====== */
#define NN_DEBUG_PRINT_WEIGHTS   0    /* 1: 训练后打印权重矩阵  0: 不打印 */
#define NN_DEBUG_PRINT_GRADIENT  0    /* 1: 训练后打印梯度      0: 不打印 */

#endif /* NN_CONFIG_H */
