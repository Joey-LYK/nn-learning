# nn_framework — 纯 C 神经网络框架

企业级分层架构，既用于教学演示，也适用于实际嵌入式工程。

## 分层架构

```
nn_framework/
│
├── core/                   ← 计算内核
│   └── nn.h/c             前向/反向传播、权重更新、API
│
├── model/                  ← 模型定义（预留）
│                           网络容器、序列化（未来从 core 拆出）
│
├── trainer/                ← 训练引擎
│   ├── trainer.h/c         训练循环（从 Task config 构建 + 训练 + 验证）
│   ├── reporter.h/c        输出报告（进度条 + 验证摘要）
│   ├── optimizer.h/c       （预留）SGD / Adam / RMSProp
│   ├── scheduler.h/c       （预留）学习率调度
│   └── loss.h/c            （预留）损失函数集合
│
├── data/                   ← 数据流水线（预留）
│   ├── dataset.h/c         （预留）数据集抽象接口
│   ├── sampler.h/c         （预留）采样器
│   └── transform.h/c       （预留）预处理/归一化
│
├── apps/                   ← 应用层（多入口）
│   ├── demo/               ← 交互式教学演示
│   │   ├── tasks/          9 个自描述 task 插件
│   │   ├── menu.h/c        菜单渲染及交互
│   │   └── main.c          入口
│   ├── dashboard/          （预留）GUI 版（基于 PainterEngine）
│   └── cli/                （预留）命令行批量训练
│
├── libs/                   ← 第三方库集中管理
│   ├── PainterEngine/      图形引擎
│   ├── logger/             （预留）日志模块
│   └── comm/               （预留）通信协议
│
├── config/                 ← 编译时配置
│   └── nn_config.h         宏开关、数据类型、Activation 枚举
│
├── platform/               ← 平台适配
│   └── nn_platform.h       控制台编码、系统 API
│
├── examples/               ← 独立示例代码
├── docs/                   ← 学习文档
├── tests/                  ← 测试
├── CMakeLists.txt
└── build.bat
```

## 可执行目标

| 目标 | 命令 | 用途 |
|------|------|------|
| `nn_demo.exe` | 交互式菜单 | 选择 9 个案例训练演示 |
| `ex_minimal.exe` | 独立运行 | 最小 API 使用示例 |
| `ex_export.exe` | 独立运行 | 模型导出/导入演示 |

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
# 方法一：直接运行 build.bat
cd nn_framework
build.bat

# 方法二：手动 cmake（中文路径下映射 N: 盘）
subst N: "E:\work\RD\神经网络"
cmake -S N:\nn_framework -B N:\nn_framework\build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build N:\nn_framework\build
subst N: /d
```

## 学习文档

学习文档已迁移到 `docs/` 目录。

> **注意**：文档中 `nn.c:{line}` 行号指向旧版单片式代码（`ui/docs/example/`），
> 新版函数位于 `core/nn.c` 中（行号已变，但函数名保持对应）。

## 依赖

- GCC (MinGW) 或任意 C99 编译器
- CMake >= 3.10（可选，可直接 gcc 编译）
