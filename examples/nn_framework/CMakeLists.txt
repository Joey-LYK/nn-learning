cmake_minimum_required(VERSION 3.10)
project(nn_framework LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

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

# ====== example: export ======
add_executable(ex_export
    ${CMAKE_SOURCE_DIR}/examples/ex_export.c
    ${NN_SRCS}
    ${CAPABILITY_SRCS}
    ${CMAKE_SOURCE_DIR}/app/task/task_sin.c
)
target_link_libraries(ex_export m)

# ====== GUI dashboard (PainterEngine) ======
option(NN_ENABLE_GUI "Build GUI dashboard (requires Windows + PainterEngine)" OFF)

if(NN_ENABLE_GUI AND WIN32)

set(PE_DIR ${CMAKE_SOURCE_DIR}/../../third_party/PainterEngine)

file(GLOB PE_CORE_SRCS   ${PE_DIR}/core/*.c)
file(GLOB PE_KERNEL_SRCS ${PE_DIR}/kernel/*.c)
file(GLOB PE_RUNTIME_SRCS
    ${PE_DIR}/runtime/PainterEngine_Runtime.c
    ${PE_DIR}/runtime/PainterEngine_Application.c
)

set(PE_WIN_C_SRCS
    ${PE_DIR}/platform/windows/px_main.c
    ${PE_DIR}/platform/windows/px_time.c
    ${PE_DIR}/platform/windows/px_file.c
    ${PE_DIR}/platform/windows/px_httpserver.c
    ${PE_DIR}/platform/windows/px_mmap.c
    ${PE_DIR}/platform/windows/px_process.c
    ${PE_DIR}/platform/windows/px_serialport.c
    ${PE_DIR}/platform/windows/px_socket.c
    ${PE_DIR}/platform/windows/PX_sockethub.c
    ${PE_DIR}/platform/windows/px_tcp.c
    ${PE_DIR}/platform/windows/px_thread.c
    ${PE_DIR}/platform/windows/px_udp.c
    ${PE_DIR}/platform/windows/px_visualos.c
)

set(PE_WIN_CPP_SRCS
    ${PE_DIR}/platform/windows/px_audio.cpp
    ${PE_DIR}/platform/windows/px_display.cpp
)

set(GUI_VIEW_SRCS
    ${CMAKE_SOURCE_DIR}/view/gui/main.c
    ${CMAKE_SOURCE_DIR}/view/gui/dash_model.c
    ${CMAKE_SOURCE_DIR}/view/gui/dash_view.c
    ${CMAKE_SOURCE_DIR}/view/gui/dash_ctrl.c
)

add_executable(nn_dashboard WIN32
    ${GUI_VIEW_SRCS}
    ${SERVICE_SRCS}
    ${CAPABILITY_SRCS}
    ${NN_SRCS}
    ${TASK_SRCS}
    ${PE_CORE_SRCS}
    ${PE_KERNEL_SRCS}
    ${PE_RUNTIME_SRCS}
    ${PE_WIN_C_SRCS}
    ${PE_WIN_CPP_SRCS}
)

target_include_directories(nn_dashboard PRIVATE
    ${PE_DIR}
    ${PE_DIR}/platform/windows
    ${PE_DIR}/runtime
)

target_link_libraries(nn_dashboard
    winmm ws2_32 comdlg32 imm32 Msimg32 gdi32 user32 shell32 ole32 d2d1 dsound
)

endif()

install(TARGETS nn_demo ex_minimal ex_export RUNTIME DESTINATION bin)
