cmake_minimum_required(VERSION 3.10)
project(nn_framework LANGUAGES C)

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

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/config
    ${CMAKE_SOURCE_DIR}/platform
)

# ====== component layer ======
set(NN_SRCS
    ${CMAKE_SOURCE_DIR}/component/nn/nn.c
)

# ====== capability layer ======
set(CAPABILITY_SRCS
    ${CMAKE_SOURCE_DIR}/capability/trainer.c
    ${CMAKE_SOURCE_DIR}/capability/reporter.c
)

# ====== app layer (tasks) ======
file(GLOB TASK_SRCS ${CMAKE_SOURCE_DIR}/app/task/task_*.c)

# ====== service layer ======
set(SERVICE_SRCS
    ${CMAKE_SOURCE_DIR}/service/training_service.c
)

# ====== view layer (cli) ======
set(CLI_VIEW_SRCS
    ${CMAKE_SOURCE_DIR}/view/cli/menu.c
)

# ====== main executable (cli mode) ======
add_executable(nn_demo
    ${CMAKE_SOURCE_DIR}/main.c
    ${CLI_VIEW_SRCS}
    ${SERVICE_SRCS}
    ${CAPABILITY_SRCS}
    ${NN_SRCS}
    ${TASK_SRCS}
)
target_link_libraries(nn_demo m)

# ====== example: minimal (component only) ======
add_executable(ex_minimal
    ${CMAKE_SOURCE_DIR}/examples/ex_minimal.c
    ${NN_SRCS}
)
target_link_libraries(ex_minimal m)

# ====== example: export (component + capability + task_sin) ======
add_executable(ex_export
    ${CMAKE_SOURCE_DIR}/examples/ex_export.c
    ${NN_SRCS}
    ${CAPABILITY_SRCS}
    ${CMAKE_SOURCE_DIR}/app/task/task_sin.c
)
target_link_libraries(ex_export m)

install(TARGETS nn_demo ex_minimal ex_export RUNTIME DESTINATION bin)
