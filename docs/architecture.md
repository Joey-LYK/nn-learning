# nn_framework 架构指南

> 本文档面向初次接触本工程的开发人员，帮助你快速理解项目结构、设计思想和扩展方式。

---

## 1. 项目是什么

nn_framework 是一个纯 C 语言实现的 **MLP 神经网络训练框架**，核心特点：

- 五层架构：component → capability → service → view → app
- 两种视图：CLI 交互式命令行 + GUI 仪表盘（PainterEngine）
- 解耦设计：view 层通过函数指针接口与 service 层通信，互不知道对方的具体类型
- 嵌入式友好：无外部依赖，nn 组件可单独提取到嵌入式平台

---

## 2. 五层架构

```
┌──────────────────────────────────────────────────────┐
│  app/task/        应用层（9 个训练任务插件）            │
│  提供 Task 实例 = 数据生成函数 + 网络配置 + 描述信息     │
├──────────────────────────────────────────────────────┤
│  view/            展示层（CLI 菜单 / GUI 仪表盘）       │
│  只依赖 ViewInterface + void* ctx，不知道 service 类型   │
├──────────────────────────────────────────────────────┤
│  service/         服务层（训练服务 + ViewInterface）     │
│  编排能力 → 业务流程，提供 step-by-step 增量训练         │
├──────────────────────────────────────────────────────┤
│  capability/      能力层（训练器、报告器、数据源接口）    │
│  组合 component 产生可独立使用的功能                     │
├──────────────────────────────────────────────────────┤
│  component/       组件层（nn 神经网络 / PainterEngine）  │
│  可替换的实现块，不依赖上层任何东西                       │
├──────────────────────────────────────────────────────┤
│  config/ + platform/   配置层 + 平台适配层               │
│  基础类型定义、编译开关、UTF-8 控制台初始化              │
└──────────────────────────────────────────────────────┘
```

**依赖方向是严格单向的**：上层可以 include 下层，下层绝不 include 上层。

```
view → service/view_interface.h → service → capability → component → config/platform
app/task → capability/data_source.h
main.c / view/gui/main.c → service + view（组装层，知道两边类型）
```

---

## 3. 目录结构

```
nn_framework/
├── main.c                          ← CLI 入口
├── CMakeLists.txt                  ← 构建脚本
├── build.bat                       ← Windows 一键编译脚本
│
├── config/
│   └── nn_config.h                 ← 激活函数枚举、数据类型、编译开关
│
├── platform/
│   └── nn_platform.h               ← 平台适配（UTF-8 控制台等）
│
├── component/
│   ├── nn/                         ← MLP 神经网络组件
│   │   ├── nn.h                    ← API 声明
│   │   └── nn.c                    ← 前向/反向/序列化实现
│   └── PainterEngine/              ← 第三方图形引擎
│
├── capability/
│   ├── data_source.h               ← Task 接口（开发新任务时实现这个）
│   ├── trainer.h / trainer.c       ← 训练能力
│   └── reporter.h / reporter.c     ← 报告能力
│
├── service/
│   ├── view_interface.h            ← View 抽象接口（解耦核心）
│   ├── training_service.h          ← 训练服务 API
│   └── training_service.c          ← 训练服务实现
│
├── app/task/                       ← 9 个任务插件
│   ├── task_sin.c                  ← sin(x) 函数拟合
│   ├── task_xor.c                  ← XOR 逻辑门
│   ├── task_projectile.c           ← 抛体运动
│   ├── task_comfort.c              ← 环境舒适度分类
│   ├── task_sevenseg.c             ← 数码管字符识别
│   ├── task_temp_comp.c            ← 传感器温度补偿
│   ├── task_battery.c              ← 电池SOC估计
│   ├── task_gesture.c              ← 手势动作识别
│   └── task_voice.c                ← 语音命令识别
│
├── view/
│   ├── cli/                        ← CLI 视图
│   │   ├── menu.h
│   │   └── menu.c
│   └── gui/                        ← GUI 视图（PainterEngine MVC）
│       ├── main.c                  ← GUI 入口
│       ├── dash_model.h / .c       ← Model
│       ├── dash_view.h / .c        ← View
│       └── dash_ctrl.h / .c        ← Controller
│
├── examples/
│   ├── ex_minimal.c                ← 最小示例（只用 nn 组件）
│   └── ex_export.c                 ← 模型导出/导入示例
│
└── docs/                           ← 学习文档
```

---

## 4. 关键数据结构

### 4.1 Task — 任务描述（`capability/data_source.h`）

每个训练任务就是一个 `Task` 实例：

```c
typedef struct {
    const char   *name;            // 显示名（如 "sin(x) 函数拟合"）
    const char   *description;     // 一句话描述
    int           input_dim;       // 输入维度
    int           output_dim;      // 输出维度
    DataGenerator generate;        // 数据生成函数指针
    TaskConfig    config;          // 网络结构和训练超参数
} Task;
```

### 4.2 TaskConfig — 训练配置（`capability/data_source.h`）

```c
typedef struct {
    int    hidden_count;                        // 隐藏层数量
    int    hidden_sizes[NN_MAX_HIDDEN_LAYERS];  // 每层神经元数
    int    hidden_act;                          // 隐藏层激活函数（Activation 枚举）
    int    output_act;                          // 输出层激活函数
    double learning_rate;
    int    epochs;
    int    samples_per_epoch;
} TaskConfig;
```

### 4.3 ViewInterface — 解耦桥梁（`service/view_interface.h`）

这是整个框架最关键的设计。它是一张函数指针表，service 层填写实现，view 层通过它调用：

```c
typedef struct {
    int         (*get_task_count)(void *ctx);
    const char* (*get_task_name)(void *ctx, int index);
    const char* (*get_task_desc)(void *ctx, int index);
    void        (*session_start)(void *ctx, int task_index);
    int         (*session_step)(void *ctx);      // 0=继续, 1=完成, -1=错误
    void        (*session_stop)(void *ctx);
    void        (*get_status)(void *ctx, ViewTrainingStatus *out);
    int         (*get_loss_history)(void *ctx, double *buf, int buf_size);
    void        (*get_train_header)(void *ctx, char *buf, int buf_size);
    void        (*get_validation)(void *ctx, char *buf, int buf_size);
} ViewInterface;
```

`view_interface.h` 不 include 任何其他项目头文件。view 层只拿到 `ViewInterface*` 和 `void* ctx`，完全不知道 service 层的具体类型。

### 4.4 ServiceContext — 内部封装（`service/training_service.c`）

service 层内部用 `ServiceContext` 包装 `svc` + `session` 指针，对外只暴露 `void*`：

```c
typedef struct {
    TrainingService *svc;
    TrainingSession *session;    // session_start 时创建，session_stop 时销毁
} ServiceContext;
```

---

## 5. 程序启动流程

### 5.1 CLI 模式（`main.c`）

```
main()
  ├── platform_console_init()          // Windows UTF-8
  ├── training_service_init(&svc)     // 创建空的任务列表
  ├── training_service_register() ×9  // 注册 9 个 Task
  ├── training_service_get_interface()// 获取 ViewInterface + ctx
  ├── cli_menu_run(&iface, ctx)       // 进入交互菜单
  │   ├── 打印菜单，用户选择任务
  │   ├── iface->session_start(ctx, index)
  │   ├── iface->get_train_header()   // 打印网络结构
  │   ├── while 循环:
  │   │   ├── iface->session_step()   // 训练一个 epoch
  │   │   └── iface->get_status()     // 显示进度
  │   ├── iface->get_validation()     // 打印验证结果
  │   └── iface->session_stop()
  ├── training_service_release_interface()
  └── free(svc.tasks)
```

### 5.2 GUI 模式（`view/gui/main.c`）

```
main()                              // 被 #define main px_main 重命名
  ├── PainterEngine_Initialize()     // 创建 1024×600 窗口
  ├── 注册 9 个 Task
  ├── training_service_get_interface()
  ├── dash_model_init(&model, &iface, ctx)
  ├── dash_view_init(&view, &model)  // 创建按钮/示波器/进度条/定时器
  ├── dash_ctrl_init(&ctrl, &model, &view)
  └── return 0;                      // PainterEngine 接管消息循环
```

用户点击按钮 → `on_button_click` → `dash_model_start()` → 16ms 定时器触发 `on_timer` → `dash_model_step()` → `dash_view_update()` 刷新 UI。

---

## 6. 构建与运行

### 6.1 环境要求

| 工具 | 要求 |
|------|------|
| CMake | >= 3.10 |
| 编译器 | MinGW-w64 GCC（推荐 13.x+） |
| 平台 | Windows（GUI 模式需要 Direct2D/DirectSound） |

### 6.2 CLI 模式（默认）

```bat
build.bat
```

或手动：

```bat
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

产出：`nn_demo.exe`（交互式训练）、`ex_minimal.exe`、`ex_export.exe`

### 6.3 GUI 模式

```bat
cmake -S . -B build_gui -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DNN_ENABLE_GUI=ON
cmake --build build_gui --target nn_dashboard
```

产出：`nn_dashboard.exe`（PainterEngine 仪表盘）

### 6.4 已知问题

**中文路径**：MinGW 的 make 无法处理非 ASCII 路径（如"神经网络"）。`build.bat` 会自动用 `subst N:` 映射虚拟盘符绕过。永久解决：把项目移到纯 ASCII 路径下。

---

## 7. 如何开发新任务

只需创建一个 `.c` 文件，实现一个 `Task` 全局实例，放到 `app/task/` 目录即可。CMakeLists.txt 会自动收集 `task_*.c` 文件。

### 模板

```c
#include "capability/data_source.h"
#include <stdlib.h>
#include <math.h>

static void generate(nn_float_t *input, nn_float_t *target) {
    // 填写 input[] 和 target[]
    // 每次调用产生一组随机训练样本
}

Task task_my_feature = {
    .name        = "我的任务",
    .description = "任务说明",
    .input_dim   = 2,
    .output_dim  = 1,
    .generate    = generate,
    .config      = {
        .hidden_count    = 2,
        .hidden_sizes    = {16, 16},
        .hidden_act      = ACT_RELU,
        .output_act      = ACT_LINEAR,
        .learning_rate   = 0.01,
        .epochs          = 5000,
        .samples_per_epoch = 50,
    },
};
```

然后在 `main.c` 和 `view/gui/main.c` 中添加注册：

```c
extern Task task_my_feature;
training_service_register(&svc, &task_my_feature);
```

### 任务配置要点

| 配置项 | 说明 |
|--------|------|
| `hidden_act` | 隐藏层激活：`ACT_RELU` / `ACT_SIGMOID` / `ACT_TANH` |
| `output_act` | 输出层激活：回归用 `ACT_LINEAR`，分类用 `ACT_SOFTMAX` |
| `output_dim` | `ACT_SOFTMAX` 时 = 类别数 |
| `samples_per_epoch` | 每 epoch 生成的样本数，影响训练速度和损失平滑度 |

---

## 8. 解耦设计详解

### 8.1 为什么需要 ViewInterface

```
没有 ViewInterface:                有 ViewInterface:

  view/gui/dash_model.h              view/gui/dash_model.h
    #include "training_service.h"       #include "view_interface.h"
    直接访问 TrainingSession 字段       只通过函数指针操作

  → view 和 service 耦死              → view 可以独立编译/测试/替换
  → 改 service 必须改 view            → 改 service 不影响 view
  → 无法支持多种 view                  → CLI/GUI/未来 Web 共用同一接口
```

### 8.2 void* ctx 的含义

`training_service_get_interface()` 返回一个不透明指针。view 层不需要知道它指向什么，只需原样传递给 ViewInterface 的每个函数。

service 层内部将其转型为 `ServiceContext*`，从中获取 `svc` 和 `session`：

```
view 层视角:     void *ctx ─────────→ 不知道，不关心
service 层视角:  (ServiceContext*)ctx → { svc, session }
```

### 8.3 组装层

`main.c` 和 `view/gui/main.c` 是唯一同时知道 service 和 view 具体类型的地方。它们负责：

1. 创建 `TrainingService`
2. 注册 tasks
3. 调用 `training_service_get_interface()` 获取 `ViewInterface + ctx`
4. 将它们注入 view 层

这就像"接线员"——把两边的插头接上，然后退场。

---

## 9. GUI MVC 模式

```
┌──────────── DashView ─────────────┐    ┌────────── DashModel ──────────┐
│  PainterEngine 控件:               │    │  通过 ViewInterface 操作:      │
│  · 9 个任务按钮                    │    │  · iface.session_start()      │
│  · Oscilloscope（loss 曲线）       │←──→│  · iface.session_step()       │
│  · ProcessBar（训练进度）           │    │  · iface.get_status()         │
│  · 3 个 Label（状态/epoch/loss）    │    │  · iface.get_loss_history()   │
│  · Timer（16ms 定时器）            │    │  缓存:                        │
│                                    │    │  · ViewTrainingStatus status  │
│  回调:                             │    │  · loss_buf[8192]             │
│  · on_button_click → 操作 Model    │    │  · is_training flag           │
│  · on_timer → model.step + update  │    └───────────────────────────────┘
└────────────────────────────────────┘
```

### 数据流

```
用户点击按钮
    │
    ▼
on_button_click()                    // dash_view.c
    ├── dash_model_stop(g_model)     // 停止上一个训练
    ├── dash_model_select(g_model, i)
    ├── dash_model_start(g_model)    // iface.session_start()
    └── 恢复定时器

on_timer() [每 16ms]                 // dash_view.c
    ├── dash_model_step(g_model)     // iface.session_step() + get_status + get_loss_history
    ├── dash_view_update(g_view, g_model)
    │   ├── 更新 Oscilloscope 数据点
    │   ├── 更新 ProcessBar 百分比
    │   └── 更新 Label 文字
    └── 如果 is_done → 暂停定时器
```

---

## 10. 构建目标一览

| 目标 | 入口 | 包含的层 | 产出 |
|------|------|---------|------|
| `nn_demo` | `main.c` | 全部（CLI view） | 交互式训练工具 |
| `ex_minimal` | `examples/ex_minimal.c` | 仅 component/nn | API 学习示例 |
| `ex_export` | `examples/ex_export.c` | component + capability | 模型序列化示例 |
| `nn_dashboard` | `view/gui/main.c` | 全部 + PainterEngine（GUI view） | 可视化仪表盘 |

---

## 11. 常见扩展场景

### 场景 A：新增一个训练任务

→ 阅读[第 7 节](#7-如何开发新任务)

### 场景 B：替换 GUI 框架（如从 PainterEngine 换成 LVGL）

1. 新建 `view/gui_v2/` 目录
2. 实现 `dash_model.h/c`（不变，因为它不依赖 PainterEngine）
3. 重写 `dash_view.h/c` 和 `main.c`（用新框架的 API 创建控件）
4. 更新 CMakeLists.txt 添加新 target

Model 层代码可以完全复用，因为 `DashModel` 只依赖 `ViewInterface`。

### 场景 C：添加新的 service 能力（如数据采集服务）

1. 在 `service/` 下新建 `data_collect_service.h/c`
2. 如果需要 view 层感知，扩展 `ViewInterface` 或创建新的 Interface
3. 在组装层（main.c）注入

### 场景 D：将 nn 组件移植到嵌入式平台

1. 复制 `component/nn/nn.c` + `nn.h` + `config/nn_config.h`
2. 在 `nn_config.h` 中设置 `NN_USE_FLOAT 1`（float 模式，节省内存）
3. 用 `nn_export` / `nn_import` 在 PC 上训练、嵌入式上推理
4. 不需要 capability/service/view 层

---

## 检查点

读到这里，你应该能回答以下问题：

1. 一个新任务需要实现哪个结构体？放在哪个目录？
2. ViewInterface 有哪 10 个函数指针？view 层如何通过它获取训练数据？
3. `main.c` 的职责是什么？为什么它同时 include service 和 view？
4. GUI 模式下，训练是在哪里被驱动的？（定时器回调）
5. 依赖方向是怎样的？`service/training_service.c` 可以 include `view/cli/menu.h` 吗？（不可以，违反依赖方向）
