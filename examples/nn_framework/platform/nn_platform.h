/**
 * nn_platform.h — 平台适配层
 *
 * 封装操作系统/编译器相关的差异，上层代码通过此头文件
 * 获得平台无关的接口。
 */

#ifndef NN_PLATFORM_H
#define NN_PLATFORM_H

#include "nn_config.h"

/* ====== 编译器兼容 ====== */
#if defined(_MSC_VER)
    #define NN_INLINE __inline
#else
    #define NN_INLINE inline
#endif

/* ====== 控制台编码设置（仅 Windows） ====== */
#if defined(_WIN32) || defined(_WIN64)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    static NN_INLINE void platform_console_init(void) {
        SetConsoleOutputCP(65001);
    }
#else
    static NN_INLINE void platform_console_init(void) {
        /* Linux/macOS 默认 UTF-8，无需设置 */
    }
#endif

#endif /* NN_PLATFORM_H */
