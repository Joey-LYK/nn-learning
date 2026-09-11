# nn_framework — 纯 C 神经网络框架

分层架构的神经网络实现，既用于教学演示，也适用于实际嵌入式工程。
本目录是本仓库教材（`textbook/`）第 17 章"从零手写神经网络库"的工程级对照实现。

## 分层架构

```
nn_framework/
│
├── core/
│   └── nn.h                ← 计算内核 API 声明（前向/反向传播、权重更新、序列化）
│
├── component/
│   └── nn/nn.c             ← 计算内核实现（对应 core/nn.h）
│
├── capability/             ← 能力层
│   ├── trainer.h/c          训练循环（Task config 构建 + 训练 + 验证）
│   ├── reporter.h/c         输出报告（进度条 + 验证摘要）
│   └── data_source.h        数据源抽象
│
├── service/
│   ├── training_service.h/c ← 服务层：训练任务编排
│   └── view_interface.h     视图接口抽象
│
├── app/
│   └── task/task_*.c       ← 9 个自描述教学任务插件
│
├── view/
│   ├── cli/menu.h/c        ← 控制台菜单视图（默认）
│   └── gui/                ← GUI 仪表盘（基于 PainterEngine，需 -DNN_ENABLE_GUI=ON）
│
├── config/
│   └── nn_config.h         ← 编译时配置：宏开关、数据类型、Activation 枚举
│
├── platform/
│   └── nn_platform.h       ← 平台适配：控制台编码、系统 API
│
├── examples/               ← 独立示例（ex_minimal.c / ex_export.c）
├── docs/architecture.md    ← 架构设计文档
├── main.c                  ← CLI 入口
├── CMakeLists.txt
└── build.bat
```

GUI 依赖的 PainterEngine 图形引擎使用仓库级共享拷贝 `third_party/PainterEngine/`
（由 CMakeLists.txt 中的 `PE_DIR` 指向），本目录不再单独携带。

## 可执行目标

| 目标 | 命令 | 用途 |
|------|------|------|
| `nn_demo.exe` | 交互式菜单 | 选择 9 个案例训练演示 |
| `ex_minimal.exe` | 独立运行 | 最小 API 使用示例 |
| `ex_export.exe` | 独立运行 | 模型导出/导入演示 |
| `nn_dashboard.exe` | GUI 窗口 | 训练过程可视化（需开启 NN_ENABLE_GUI） |

## 9 个教学案例

| # | 名称 | 类型 | 应用方向 |
|---|------|------|---------|
| 1 | sin(x) 函数拟合 | 回归 | 函数逼近入门 |
| 2 | XOR 逻辑门 | 分类 | 非线性分类基础 |
| 3 | 抛体运动 | 回归 | 仿真数据 + 归一化 |
| 4 | 环境舒适度分类 | 分类 | 传感器数据分类 |
| 5 | 数码管字符识别 | 分类 | 视觉模式识别 |
| 6 | 传感器温度补偿 | 回归 | NTC 非线性校正 |
| 7 | 电池 SOC 估计 | 回归 | BMS 核心功能 |
| 8 | 手势动作识别 | 分类 | 时序传感器分类 |
| 9 | 语音命令识别 | 分类 | 音频特征分类 |

## 构建方法

```bash
# 方法一：直接运行 build.bat（Windows + MinGW）
cd examples/nn_framework
build.bat

# 方法二：手动 cmake（中文路径下建议先映射 N: 盘规避 MinGW 路径问题）
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 构建 GUI 仪表盘（Windows）
cmake -S . -B build_gui -G "MinGW Makefiles" -DNN_ENABLE_GUI=ON
cmake --build build_gui
```

## 配套学习文档

学习文档位于仓库根目录 [`docs/`](../../docs/)，系统教材位于 [`textbook/`](../../textbook/)。

> 文档中 `nn.c:{line}` 行号指向旧版单片式代码，新版实现位于
> `component/nn/nn.c`（函数名保持对应，行号有变化）。

## 依赖

- GCC (MinGW) 或任意 C99 编译器
- CMake >= 3.10（可选，也可直接 gcc 编译）
