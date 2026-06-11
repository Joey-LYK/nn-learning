cmake_minimum_required(VERSION 3.10)
project(nn_framework LANGUAGES C)

# ====== 编译选项 ======
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0 -DNN_DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O2)
endif()

# ====== 公共头文件路径 ======
# 双模式支持：qualified path ("core/nn.h") + short name ("nn_config.h")
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/config
    ${CMAKE_SOURCE_DIR}/platform
)

# ====== 收集源文件 ======
# core 层 — 计算内核（张量、算子、自动微分）
set(CORE_SRCS
    ${CMAKE_SOURCE_DIR}/core/nn.c
)

# trainer 层 — 训练引擎（优化器、调度器、loss、训练循环、报告）
set(TRAINER_SRCS
    ${CMAKE_SOURCE_DIR}/trainer/trainer.c
    ${CMAKE_SOURCE_DIR}/trainer/reporter.c
)

# apps/demo 层 — 交互式教学演示（任务 + 菜单）
file(GLOB DEMO_TASK_SRCS ${CMAKE_SOURCE_DIR}/apps/demo/tasks/task_*.c)
set(DEMO_SRCS
    ${CMAKE_SOURCE_DIR}/apps/demo/main.c
    ${CMAKE_SOURCE_DIR}/apps/demo/menu.c
    ${DEMO_TASK_SRCS}
)

# ====== 主可执行文件：交互式训练演示 ======
add_executable(nn_demo
    ${DEMO_SRCS}
    ${TRAINER_SRCS}
    ${CORE_SRCS}
)
target_link_libraries(nn_demo m)

# ====== 示例：最小示例（只用 core） ======
add_executable(ex_minimal
    ${CMAKE_SOURCE_DIR}/examples/ex_minimal.c
    ${CORE_SRCS}
)
target_link_libraries(ex_minimal m)

# ====== 示例：模型导出（core + trainer + task_sin） ======
add_executable(ex_export
    ${CMAKE_SOURCE_DIR}/examples/ex_export.c
    ${CORE_SRCS}
    ${TRAINER_SRCS}
    ${CMAKE_SOURCE_DIR}/apps/demo/tasks/task_sin.c
)
target_link_libraries(ex_export m)

# ====== 安装 ======
install(TARGETS nn_demo ex_minimal ex_export RUNTIME DESTINATION bin)
