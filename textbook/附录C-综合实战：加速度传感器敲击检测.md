# 附录C 综合实战：加速度传感器的敲击检测

> 本附录学习目标：
> 1. 能走通一个真实传感器任务的全流程：采数 → 打标 → 分窗提特征 → 训练 → 导出 → C 端推理
> 2. 能解释"分窗提特征"在做什么：把一段连续波形变成一串定长的数字
> 3. 能用第 17 章的手写库完成一次三类分类任务，并按类别评估效果
> 4. 能把训练好的权重导出成文本文件，用不到百行的 C 程序在另一端复现同样的预测
> 5. 能说出这个流程落到真实设备上时最容易翻车的四个坑

## 从一个例子开始

现在的智能手表大多有一个隐藏技能：**双击表盘截屏**。你用指节敲两下表盘，它就截图；而你洗手、打字、跑步时手臂的千万次晃动，它都无动于衷。

怎么做到的？手表里有一个**加速度传感器**（accelerometer），它每时每刻报告三个方向上的加速度（单位是 g，重力加速度的倍数）。指节敲击是一次尖锐的"脉冲 + 振铃"；而误触——手表磕到桌面、打字时手腕抖动——要么太轻柔，要么太拖沓。**这些差别藏在波形的形状里**，本附录的任务就是训练一个小网络把它们分开：

- **类别 0：误触**——磕碰、拍打、走路震动，幅度小、起伏慢
- **类别 1：轻敲**——有意的轻轻一敲，尖锐短促
- **类别 2：重敲**——用力敲，幅度大、振铃明显

分出三类，"是敲击还是误触"和"力度多大"就一起回答了。这也是很多嵌入式课程的经典结课作业，本附录带你从零到一完成它——**用到的每一件工具，都是正文里亲手造过的**。

## C.0 流水线总览

先看全景。整个任务分六步，前四步在电脑上完成，最后两步交付给设备：

```mermaid
flowchart LR
    A["① 采数<br>录制三类加速度波形"] --> B["② 打标<br>给每条数据记上类别"]
    B --> C["③ 分窗提特征<br>波形 → 6 个数字"]
    C --> D["④ 训练<br>用第 17 章的库"]
    D --> E["⑤ 导出<br>权重写进文本文件"]
    E --> F["⑥ C 端推理<br>设备上复现同样的预测"]
```

一句话记住分工：**训练在电脑上（Python），推理在设备上（C）**。所以本附录的 C 代码只出现在两个真正要跑在设备端的环节——特征提取（C.3 的对照块）和推理程序（C.5）。采数、打标、训练是一次性的电脑工作，用 Python 顺手即可。

所有代码只用 Python 标准库和一个传感器模拟器，**没有传感器的读者也能完整跑通**；有真实设备的读者，只需把第①步的模拟数据换成你的录制文件，其余步骤一字不改。

## C.1 第一步：采数——没有数据就没有网络

### 传感器怎么配置

以最常见的MEMS加速度传感器为例，两个旋钮要拧对：

- **量程（range）**：±4g。量程太小，重敲会"削顶"（超出量程直接截断）；太大则小信号分辨率不足
- **采样率（sample rate）**：100 Hz，即每秒记录 100 个点。敲击的振铃通常持续几十毫秒，100 Hz 足够看清它

每次敲击的波形约 200 ms，采样率 100 Hz 时就是 **20 个采样点**，每个点有 x、y、z 三个数。所以一条数据 = 60 个数字 + 1 个标签。

### 采集纪律

真实采数时最容易偷懒的三件事，恰恰决定成败：

1. **每类至少上百条，且要"多样"**。轻敲不能一个人用同一种姿势敲 100 下——要换手、换位置、换力度，否则网络学的是"你的手"而不是"敲击"
2. **误触要专门去"采"**。误触不会自己出现，你得带着设备打字、走路、放包里磕碰，把真实的干扰录下来。漏采误触，网络就会把一切晃动都当敲击
3. **顺序随机化**。不要连着录完所有轻敲再录重敲——设备温度、电池状态都在缓慢变化，按顺序录会把"时间"泄漏进数据（第 14 章的数据泄漏问题）

### 用模拟器代替传感器

下面这段代码生成**模拟的传感器数据**：误触用"幅度小、起伏慢的高斯鼓包"模拟，敲击用"陡峭起振 + 衰减振铃"模拟（`abs(sin)` 刻画结构振荡的幅值包络），再沿随机方向分解到三个轴、叠加噪声。生成的同时就打好了标（类别 0/1/2），写进 `tap_events.csv`。

```python
# -*- coding: utf-8 -*-
"""附录C 块1：模拟采数 + 打标 → tap_events.csv"""
import math
import random

random.seed(7)

SR = 100    # 采样率 100 Hz
WIN = 20    # 每个事件窗 20 个样本 = 200 ms
N_PER_CLASS = 120   # 每类 120 条


def random_dir():
    """随机单位向量：每次敲击的方向各不相同"""
    theta = random.uniform(0, 2 * math.pi)
    z = random.uniform(-0.8, 0.8)
    r = math.sqrt(max(0.0, 1 - z * z))
    return [r * math.cos(theta), r * math.sin(theta), z]


def make_mistouch():
    """误触：幅度小、起伏慢的'鼓包'"""
    amp = random.uniform(0.12, 0.25)
    center = random.uniform(8, 12)
    width = random.uniform(3.5, 5.5)
    mag = [amp * math.exp(-((i - center) ** 2) / (2 * width ** 2))
           for i in range(WIN)]
    return mag, 0.02


def make_tap(heavy):
    """敲击：陡峭起振 + 衰减振铃"""
    amp = random.uniform(0.9, 1.6) if heavy else random.uniform(0.4, 0.8)
    freq = random.uniform(20, 30)     # 振铃频率 Hz
    decay = random.uniform(3.0, 5.0)  # 衰减常数（样本数）
    delay = random.randint(2, 4)      # 起振前的静止
    mag = []
    for i in range(WIN):
        t = i - delay
        if t < 0:
            mag.append(0.0)
        else:
            mag.append(amp * math.exp(-t / decay)
                       * abs(math.sin(2 * math.pi * freq * t / SR)))
    return mag, 0.03


def save_event(path, label, mag, noise):
    """沿随机方向把幅值分解到 3 个轴，加噪声后写一行 CSV"""
    dx, dy, dz = random_dir()
    cells = [str(label)]
    for i in range(WIN):
        ax = mag[i] * dx + random.gauss(0, noise)
        ay = mag[i] * dy + random.gauss(0, noise)
        az = mag[i] * dz + random.gauss(0, noise)
        cells += [f"{ax:.4f}", f"{ay:.4f}", f"{az:.4f}"]
    path.append(",".join(cells))


rows = []
for _ in range(N_PER_CLASS):
    mag, noise = make_mistouch()
    save_event(rows, 0, mag, noise)          # 类别 0：误触
for _ in range(N_PER_CLASS):
    mag, noise = make_tap(heavy=False)
    save_event(rows, 1, mag, noise)          # 类别 1：轻敲
for _ in range(N_PER_CLASS):
    mag, noise = make_tap(heavy=True)
    save_event(rows, 2, mag, noise)          # 类别 2：重敲

random.shuffle(rows)                         # 顺序随机化
with open("tap_events.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")
print(f"tap_events.csv: {len(rows)} 条事件，每条 {WIN} 个采样点 × 3 轴")
```

**预期输出**：

```
tap_events.csv: 360 条事件，每条 20 个采样点 × 3 轴
```

CSV 每行的格式是：`标签, x0, y0, z0, x1, y1, z1, ..., x19, y19, z19`——共 1 + 60 个数。真实项目里，这行 CSV 就对应"一次事件的一段录制 + 人工记录的标签"，格式相同，只是数字来自传感器。

## C.2 第二步：打标

模拟器里标签是生成时顺手写的；真实采数时，打标通常有两种做法：

- **边采边标**：脚本控制采集流程——屏幕提示"请轻敲 10 下"，按回车开始录制，录完自动记下标签。省事，但"提示的"动作往往不自然
- **先采后标**：连续录一整段，事后用工具回放、切分、逐条标。更真实，但多一道人工工序

无论哪种，都要记住第 3 章说过的纪律：**标注质量决定上限**。把"重敲"随手标成"轻敲"的脏标签，网络会忠实地学进去。拿不准的样本宁可丢弃，也不要猜一个标签塞进去。

## C.3 第三步：分窗提特征——把波形变成数字

原始波形一条 60 个数，而且不同事件的波形"长得"各种各样。直接把 60 个数喂给网络当然可以，但更好的做法是先问：**人类看这段波形时，看的是哪几个量？**

- 敲得多重？看**峰值**（幅值最大处）
- 持续多久、动静多大？看**均值**和**有效值**（RMS，均方根）
- 起得多快？敲击是"陡"的，误触是"缓"的——看**陡峭度**（相邻采样点差值的最大值）
- 波形是否瘦削？看**过半样本数**（幅值超过峰值一半的采样点个数：敲击瘦、误触胖）
- 方向信息？看 **z 轴均值**（很多设备只对某个方向的敲击感兴趣）

60 个数浓缩成 6 个数，这一步叫**特征提取（feature extraction）**。第 16 章说"数据决定效果上限"，特征工程就是数据工作的核心一击——**好的特征能让网络变简单**。

```python
# -*- coding: utf-8 -*-
"""附录C 块2：分窗提特征 → tap_features.csv"""
import math

WIN = 20
FEATURE_NAMES = ["峰值", "均值", "有效值", "陡峭度", "过半样本数", "z轴均值"]


def extract(ax, ay, az):
    """一段窗口（3 轴 × 20 点）→ 6 个特征"""
    n = len(ax)
    mag = [math.sqrt(ax[i] ** 2 + ay[i] ** 2 + az[i] ** 2) for i in range(n)]
    peak = max(mag)                                             # 峰值
    mean = sum(mag) / n                                         # 均值
    rms = math.sqrt(sum(v * v for v in mag) / n)                # 有效值
    sharp = max(abs(mag[i + 1] - mag[i]) for i in range(n - 1)) # 陡峭度
    above = sum(1 for v in mag if v > peak / 2)                 # 过半样本数
    z_mean = sum(az) / n                                        # z 轴均值
    return [peak, mean, rms, sharp, above, z_mean]


out = []
for line in open("tap_events.csv", encoding="utf-8"):
    parts = line.strip().split(",")
    label = int(parts[0])
    vals = [float(v) for v in parts[1:]]
    ax = vals[0::3]                     # 每 3 个数取 1 个：x 轴序列
    ay = vals[1::3]                     # y 轴序列
    az = vals[2::3]                     # z 轴序列
    feats = extract(ax, ay, az)
    out.append(str(label) + "," + ",".join(f"{v:.4f}" for v in feats))

with open("tap_features.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print(f"tap_features.csv: {len(out)} 行 × {len(FEATURE_NAMES)} 个特征")

# 每类抽一条看特征，直观感受三类分得开不开
for label, name in [(0, "误触"), (1, "轻敲"), (2, "重敲")]:
    for line in out:
        if line.startswith(str(label) + ","):
            vals = [float(v) for v in line.split(",")[1:]]
            pretty = ", ".join(f"{n}={v:.3f}" for n, v in zip(FEATURE_NAMES, vals))
            print(f"{name}: {pretty}")
            break
```

**预期输出**：

```
tap_features.csv: 360 行 × 6 个特征
误触: 峰值=0.267, 均值=0.123, 有效值=0.150, 陡峭度=0.086, 过半样本数=8.000, z轴均值=-0.027
轻敲: 峰值=0.373, 均值=0.081, 有效值=0.116, 陡峭度=0.305, 过半样本数=2.000, z轴均值=-0.012
重敲: 峰值=0.862, 均值=0.141, 有效值=0.241, 陡峭度=0.796, 过半样本数=1.000, z轴均值=-0.010
```

看这三行，特征的价值一目了然：

| 特征 | 误触 | 轻敲 | 重敲 | 单独看能否区分 |
|------|------|------|------|----------------|
| 峰值 | 0.267 | 0.373 | 0.862 | 重敲能分，轻敲和误触接近 |
| 陡峭度 | 0.086 | 0.305 | 0.796 | 误触和敲击分得很开 |
| 过半样本数 | 8 | 2 | 1 | 误触"胖"，敲击"瘦" |

**没有任何单一特征能独揽全局，但六个特征拼起来，三类被围得水泄不通**——这就是特征工程的威力。真实项目里，提出一组好特征往往比调网络省力得多。

<details>
<summary>🐘 C 语言对照</summary>

*这一块值得给 C 对照，因为**设备端最终也要用 C 算同样的特征**——传感器中断进来，把窗口缓冲区交给这个函数即可。逐行对照可以看到：Python 的切片 `vals[0::3]` 对应 C 的下标运算 `i * 3`，列表推导对应 for 循环。*

```c
/* feat_extract.c —— 与块2的 extract() 逐行对应的 C 版本 */
#include <stdio.h>
#include <math.h>

#define WIN 20

void extract(const double ax[], const double ay[], const double az[],
             double feat[]) {
    double mag[WIN];                        /* 合成幅值：三轴的向量和 */
    for (int i = 0; i < WIN; i++) {
        mag[i] = sqrt(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i]);
    }
    double peak = mag[0], sum = 0, sq = 0, sharp = 0, z_sum = 0;
    for (int i = 0; i < WIN; i++) {
        sum += mag[i];
        sq += mag[i] * mag[i];
        z_sum += az[i];
        if (mag[i] > peak) peak = mag[i];
    }
    for (int i = 0; i + 1 < WIN; i++) {
        double d = fabs(mag[i + 1] - mag[i]);
        if (d > sharp) sharp = d;           /* 陡峭度：相邻差值的最大值 */
    }
    int above = 0;
    for (int i = 0; i < WIN; i++) {
        if (mag[i] > peak / 2) above++;     /* 过半样本数 */
    }
    feat[0] = peak;
    feat[1] = sum / WIN;                    /* 均值 */
    feat[2] = sqrt(sq / WIN);               /* 有效值 RMS */
    feat[3] = sharp;
    feat[4] = (double)above;
    feat[5] = z_sum / WIN;                  /* z 轴均值 */
}

int main(void) {
    /* 喂一个 20 点的慢速高斯鼓包，模拟一次误触 */
    double ax[WIN] = {0}, ay[WIN] = {0}, az[WIN] = {0};
    for (int i = 0; i < WIN; i++) {
        az[i] = 0.2 * exp(-((i - 10.0) * (i - 10.0)) / 18.0);
    }
    double feat[6];
    extract(ax, ay, az, feat);
    const char *names[6] = {"峰值", "均值", "有效值", "陡峭度", "过半样本数", "z轴均值"};
    for (int j = 0; j < 6; j++) {
        printf("%s = %.3f\n", names[j], feat[j]);
    }
    return 0;
}
```

main 里喂的是一个宽 5.5 点、高 0.2g 的高斯鼓包（模拟误触），**预期输出**——和正文误触样例的特征形态一致（峰值小、陡峭度小、过半样本多）：

```
峰值 = 0.200
均值 = 0.075
有效值 = 0.103
陡峭度 = 0.039
过半样本数 = 7.000
z轴均值 = 0.075
```

</details>

## C.4 第四步：用第 17 章的库训练

万事俱备。先把第 17 章"库的代码"一节里的 `Layer` 类和 `Network` 类两段代码存成 `mininn.py`（与本附录脚本放同一目录），然后：

1. 按类划分数据：每类 70% 训练、15% 验证、15% 测试（252 / 54 / 54 条）
2. **归一化统计量只从训练集算**（第 16 章的铁律，测试集要"装作没见过"）
3. 网络结构 `Network([6, 8, 3])`：6 个特征进，8 个 ReLU 隐藏神经元，3 个 sigmoid 输出
4. 标签用 one-hot：轻敲 = `[0, 1, 0]`。预测时取输出最大的那类（argmax）

```python
# -*- coding: utf-8 -*-
"""附录C 块3：划分 → 归一化 → 训练 → 按类评估"""
import random
from mininn import Network

random.seed(42)

# ---- 按类划分 70/15/15，保持每类比例一致 ----
by_class = {0: [], 1: [], 2: []}
for line in open("tap_features.csv", encoding="utf-8"):
    parts = line.strip().split(",")
    by_class[int(parts[0])].append([float(v) for v in parts[1:]])

train, val, test = [], [], []
for label, feats in by_class.items():
    n = len(feats)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    for i, f in enumerate(feats):
        y = [1.0 if k == label else 0.0 for k in range(3)]   # one-hot 标签
        if i < n_train:
            train.append((f, y, label))
        elif i < n_train + n_val:
            val.append((f, y, label))
        else:
            test.append((f, y, label))
print(f"划分：训练 {len(train)} / 验证 {len(val)} / 测试 {len(test)}")

# ---- 归一化：统计量只用训练集（第 16 章的铁律）----
N_F = 6
fmin = [min(s[0][j] for s in train) for j in range(N_F)]
fmax = [max(s[0][j] for s in train) for j in range(N_F)]

def norm(feats):
    return [(v - lo) / (hi - lo) for v, lo, hi in zip(feats, fmin, fmax)]

# ---- 训练：6 特征 → 8 隐藏(ReLU) → 3 输出(sigmoid) ----
net = Network([6, 8, 3])
net.train([(norm(f), y) for f, y, _ in train], epochs=400, lr=0.5, log_every=50)

def predict(feats):
    out = net.forward(norm(feats))
    return out.index(max(out))          # 三个输出里最大的当答案

def report(name, data):
    correct = [0, 0, 0]
    total = [0, 0, 0]
    for f, _, label in data:
        total[label] += 1
        if predict(f) == label:
            correct[label] += 1
    accs = ["{:.0f}%".format(100 * c / t) for c, t in zip(correct, total)]
    print(f"{name}: 总准确率 {100*sum(correct)/sum(total):.1f}%  "
          f"(误触 {accs[0]}  轻敲 {accs[1]}  重敲 {accs[2]})")

report("验证集", val)
report("测试集", test)
```

**预期输出**：

```
划分：训练 252 / 验证 54 / 测试 54
轮次  50  平均损失 0.0112
轮次 100  平均损失 0.0092
轮次 150  平均损失 0.0055
轮次 200  平均损失 0.0069
轮次 250  平均损失 0.0049
轮次 300  平均损失 0.0049
轮次 350  平均损失 0.0037
轮次 400  平均损失 0.0040
验证集: 总准确率 98.1%  (误触 100%  轻敲 100%  重敲 94%)
测试集: 总准确率 98.1%  (误触 100%  轻敲 100%  重敲 94%)
```

54 条测试样本里只错 1 条（重敲 94% = 17/18）。这个"不完美"值得停下来讨论两点：

1. **为什么要按类报告，而不只看总准确率？** 假设误触识别率只有 80%——总准确率照样好看，但产品每 5 次磕碰就误唤醒一次，用户会疯掉。**不同类别的错误代价不同**：误触漏检（把误触当敲击）通常比敲击漏检严重得多。按类看，才知道错误落在哪
2. **验证集和测试集成绩几乎一样（98.1%）**，说明没有明显过拟合（第 14 章的判据）——数据够、模型小（只有 6×8+8+8×3+3 = 83 个参数），网络"背不动"答案

注意输出层用的是每类一个 sigmoid 而不是第 13 章的 softmax——这是第 17 章库的原样能力，够完成本任务；softmax + 交叉熵是更标准的做法，正是第 17 章习题 5 的改造方向。

## C.5 第五步：导出权重，C 端推理

训练好的网络，全部家当就是归一化统计量（fmin、fmax 各 6 个数）加每层的权重和偏置。把它们写成一个文本文件——**模型文件总共只有 1.1 KB**，这就是"小模型"三个字的具体分量。

接在块 3 的同一个文件里继续写：

```python
# -*- coding: utf-8 -*-
"""附录C 块4：导出权重（接块3的同一个文件继续写）"""

ACT_CODE = {"relu": 0, "sigmoid": 1}
lines = ["TAPNET-V1"]
lines.append(" ".join(f"{v:.6f}" for v in fmin))          # 归一化下限
lines.append(" ".join(f"{v:.6f}" for v in fmax))          # 归一化上限
lines.append(str(len(net.layers)))
for layer in net.layers:
    n_out = len(layer.b)
    lines.append(f"{len(layer.w[0])} {n_out} {ACT_CODE[layer.activation]}")
    for j in range(n_out):
        lines.append(" ".join(f"{w:.8f}" for w in layer.w[j]))   # 权重逐行
    lines.append(" ".join(f"{b:.8f}" for b in layer.b))          # 再写偏置
with open("tap_model.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

# 测试集原始特征也落盘，供 C 端自行归一化后推理对账
with open("tap_test.txt", "w", encoding="utf-8") as f:
    f.write(str(len(test)) + "\n")
    for feats, _, label in test:
        f.write(str(label) + " " + " ".join(f"{v:.6f}" for v in feats) + "\n")
print("已导出 tap_model.txt（权重+归一化统计量）和 tap_test.txt（测试样本）")
```

**预期输出**：

```
已导出 tap_model.txt（权重+归一化统计量）和 tap_test.txt（测试样本）
```

文件格式一望可知：第一行是版本标识（第 17 章的工程习惯，防止误加载），接着是归一化统计量、层数，然后每层是"输入数 输出数 激活类型"一行头 + 权重逐行 + 偏置一行。**注意写的顺序：每层先写完全部权重行，再写偏置**——等下 C 端读的时候必须严格按同样顺序。

然后是本附录的主角：不到 90 行的 C 推理程序。它不做训练、不求梯度，只做三件事——读模型、读样本、前向传播：

```c
/* tap_infer.c —— C 端推理：读模型 + 读样本 → 逐条前向 → 按类报告准确率
   编译：gcc -std=c99 -Wall tap_infer.c -o tap_infer -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_N 16
#define MAX_L 4
#define N_F   6        /* 特征个数 */

typedef struct {
    int n_in, n_out, act;          /* act：0=ReLU，1=sigmoid */
    double w[MAX_N][MAX_N];
    double b[MAX_N];
} Layer;

double activate(double z, int act) {
    if (act == 0) return z > 0 ? z : 0.0;          /* ReLU */
    return 1.0 / (1.0 + exp(-z));                  /* sigmoid */
}

int main(void) {
    /* ---- 1. 读模型文件：归一化统计量 + 每层权重 ---- */
    FILE *f = fopen("tap_model.txt", "r");
    if (f == NULL) { printf("找不到 tap_model.txt\n"); return 1; }
    char tag[32];
    fscanf(f, "%31s", tag);
    if (strcmp(tag, "TAPNET-V1") != 0) {           /* 版本校验 */
        printf("模型文件格式不对\n"); return 1;
    }
    double fmin[N_F], fmax[N_F];
    for (int j = 0; j < N_F; j++) fscanf(f, "%lf", &fmin[j]);
    for (int j = 0; j < N_F; j++) fscanf(f, "%lf", &fmax[j]);

    int nl;
    fscanf(f, "%d", &nl);
    static Layer layers[MAX_L];
    for (int k = 0; k < nl; k++) {
        Layer *l = &layers[k];
        fscanf(f, "%d %d %d", &l->n_in, &l->n_out, &l->act);
        for (int j = 0; j < l->n_out; j++)          /* 先读完全部权重行 */
            for (int i = 0; i < l->n_in; i++) fscanf(f, "%lf", &l->w[j][i]);
        for (int j = 0; j < l->n_out; j++)          /* 再读偏置向量 */
            fscanf(f, "%lf", &l->b[j]);
    }
    fclose(f);

    /* ---- 2. 逐条读测试样本：归一化 → 前向 → argmax ---- */
    FILE *tf = fopen("tap_test.txt", "r");
    if (tf == NULL) { printf("找不到 tap_test.txt\n"); return 1; }
    int n;
    fscanf(tf, "%d", &n);

    int correct[3] = {0, 0, 0}, total[3] = {0, 0, 0};
    for (int s = 0; s < n; s++) {
        int label;
        double x[N_F], buf_a[MAX_N], buf_b[MAX_N];   /* 缓冲要装下最宽的层 */
        fscanf(tf, "%d", &label);
        double *in = buf_a, *out = buf_b;   /* 双缓冲：输出不能覆盖还要用的输入 */
        for (int j = 0; j < N_F; j++) {
            fscanf(tf, "%lf", &x[j]);
            in[j] = (x[j] - fmin[j]) / (fmax[j] - fmin[j]);  /* 归一化 */
        }
        for (int k = 0; k < nl; k++) {             /* 前向传播 */
            const Layer *l = &layers[k];
            for (int j = 0; j < l->n_out; j++) {
                double z = l->b[j];
                for (int i = 0; i < l->n_in; i++) z += l->w[j][i] * in[i];
                out[j] = activate(z, l->act);
            }
            double *t = in; in = out; out = t;     /* 本层输出变下层输入 */
        }
        int pred = 0;   /* argmax：注意最后一次交换后，最终输出在 in 里 */
        for (int j = 1; j < layers[nl - 1].n_out; j++) {
            if (in[j] > in[pred]) pred = j;
        }
        total[label]++;
        if (pred == label) correct[label]++;
    }
    fclose(tf);

    int sum_c = correct[0] + correct[1] + correct[2];
    int sum_t = total[0] + total[1] + total[2];
    printf("C 端测试集: 总准确率 %.1f%%  (误触 %d%%  轻敲 %d%%  重敲 %d%%)\n",
           100.0 * sum_c / sum_t,
           100 * correct[0] / total[0],
           100 * correct[1] / total[1],
           100 * correct[2] / total[2]);
    return 0;
}
```

编译运行（模型文件和测试文件放在同一目录）：

```
gcc -std=c99 -Wall tap_infer.c -o tap_infer -lm
./tap_infer
```

**预期输出**：

```
C 端测试集: 总准确率 98.1%  (误触 100%  轻敲 100%  重敲 94%)
```

**与 Python 端的测试报告逐位一致**。同一条数据，电脑上训练的网络和设备上的 C 程序给出同一个答案——这就是"部署"两个字的全部含义。仓库 `examples/nn_framework/` 里的 `nn_export`/`nn_import` 用二进制格式做了同一件事，是本节思路的工程版。

<details>
<summary>🐘 语言差异提示（本块就是 C，反过来给 Python 读者）</summary>

*如果你只读过本书的 Python，这段 C 里陌生的东西只有四样：`FILE *` 和 `fscanf` 是"打开文件、按格式读数"；`double w[16][16]` 是固定尺寸的二维数组；`layers[1].w[2][i]` 读作"第 1 层第 2 个神经元的第 i 个权重"；其余的 for、if、函数调用与 Python 一一对应。前向传播那段双重循环，就是第 7 章矩阵乘法的逐元素写法。*

</details>

## C.6 从演示到产品：设备端还差什么

真实设备上的完整工作循环是这样的：

1. **平时休眠**：传感器以低功耗持续采样，固件只盯一个简单指标——合成幅值是否超过阈值（比如 0.3g）。这一步不用网络
2. **触发截窗**：超过阈值就把前后共 200 ms 的数据存进缓冲区——这就是"分窗"，C.3 的 `extract()` 在设备端算出 6 个特征
3. **前向推理**：归一化（用模型文件里的统计量）→ 两层前向 → argmax。整个网络只有 **83 次乘加**，对现代单片机是几微秒的事
4. **执行动作**：判定为敲击就触发；连续检测到两次轻敲就是"双击"

顺带回答作业里"分析力度"的部分：**力度本身不需要网络**——峰值（特征 1）直接就是力度的物理量，可以拿去做分级或显示；网络的职责是"这算不算一次有意的敲击、属于哪一类"。让合适的工具干合适的活，是工程师的判断力。

最后是验收清单，四个最容易翻车的坑：

| 坑 | 症状 | 解法 |
|------|------|------|
| 归一化统计量丢失 | 设备端精度骤降、结果飘忽 | fmin/fmax 必须随模型一起导出（本附录已做）——用训练集统计量，漏了就是反向的数据泄漏 |
| 训练/部署采样率不一致 | 真机上几乎全错，测试集上却很好 | 采样率写进模型文件并校验；变更采样率必须重新采数训练 |
| 实验室与现场分布不同 | 实验室 98%，现场 60% | 现场环境（佩戴方式、温度、振动源）要覆盖进训练数据；上线后持续收集难例 |
| 只看总准确率 | 总分漂亮，误触漏检一堆 | 按类报告；把"误触识别率"当作第一指标 |

## 常见误区

**误区：直接把 60 个原始采样点喂给网络，比提特征"更高级"。**
真相：本任务的原始输入 60 维、每类仅 120 条样本，直接喂会让网络在第 14 章说过"参数量 > 数据信息量"的陷阱里过拟合。特征提取是用对数据的理解换网络容量——60 维压成 6 维后，83 个参数的模型就足够了。数据多到千万级时，"端到端学特征"才划算（那是第 18 章 CNN 的思路）。

**误区：训练用了归一化，部署时忘了；或者反过来。**
真相：归一化是模型的一部分。C 端推理前必须用**训练集统计量**做同样的变换——本附录把它们写进了模型文件就是这个原因。训练归一化、推理不归一化（或用了测试集的统计量），输入分布完全对不上，预测必然错乱。

**误区：自定义模型文件格式时，"能读出来"就等于"读对了"。**
真相：本附录的格式里，权重和偏置的**写序、读序必须严格一致**。写的时候"每层先写完全部权重行再写偏置"，读的时候却"读完一行权重就读一个偏置"，程序不会报任何错——只是把别人的数安到自己的参数头上，精度从 98% 悄悄掉到 57%。这类错误不崩、不报错、只污染结果，排查手段只有一个：像 C.5 那样拿一条样本在两端逐值对账。

**误区：C 端缓冲区按"特征个数 6"开就行，反正输入就 6 个数。**
真相：前向传播的中间缓冲要装下**最宽的那一层**（本例是隐藏层的 8 个），按输入开 6 个元素会在隐藏层写出界——栈上的越界写会悄悄破坏相邻变量，症状是"结果时对时错"。缓冲区尺寸应该由模型结构决定（读模型时校验 `n_out <= MAX_N`），而不是由输入决定。

## 章末习题

1. 把块 1 中轻敲的幅度范围从 `0.4~0.8` 改成 `0.5~0.7`（更"标准"的轻敲），重跑全流程，测试集准确率是升还是降？结合"数据多样性"讨论这个结果。
2. 块 2 的 6 个特征里，删掉"陡峭度"再训练，误触和轻敲还分得开吗？哪两个特征补上了它留下的空缺？
3. 把网络改成 `Network([6, 4, 3])`（隐藏层砍半），准确率掉多少？再改成 `[6, 32, 3]` 呢？用第 14 章的话解释两种现象。
4. C.5 的模型文件把激活类型存成了数字（0/1）。如果有一天要新增 tanh，格式要怎么升级才不破坏旧文件？（提示：回忆第 17 章里 `format` 版本标识的作用。）
5. （思考题）真实手表上，"轻敲"和"重敲"往往不需要区分——都执行同一个动作。既然如此，训练成三类还有什么价值？（提示：想想训练数据的采集难度，以及一个"误触 vs 敲击"二分类网络的输出层 sigmoid 阈值该怎么用力度去调。）

## 小结

- 一个完整的嵌入式机器学习任务 = 采数 → 打标 → 分窗提特征 → 训练 → 导出 → C 端推理，前四步在电脑上，后两步在设备上
- 特征工程是用对数据的理解换网络容量：60 维波形压成 6 维特征，83 个参数的小网络就能做到 98% 的按类准确率
- 按类评估比总准确率诚实：不同类别的错误代价不同，本任务里"误触识别率"是第一指标
- 部署 = 归一化统计量 + 权重 + 激活类型的完整交付（全部信息 1.1 KB），C 端前向只有 83 次乘加
- 自定义格式的读写顺序、随模型走归一化统计量、按最宽层开缓冲——三个工程细节，任何一个出错都不报错，只污染结果

到这里，这本书真正画完了圆：第 1~5 章的数学、第 6~9 章的直觉、第 10~13 章的原理、第 14~17 章的手艺，在这个小任务里各就各位。剩下的——换你的传感器、你的数据、你的创意。
