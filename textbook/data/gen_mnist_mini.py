# -*- coding: utf-8 -*-
"""生成附录D 使用的 MNIST 平衡子集 mnist_mini.gz。

输入:MNIST 官方 IDX 原始文件(gzip 压缩),可从
    https://ossci-datasets.s3.amazonaws.com/mnist/
下载 train-images-idx3-ubyte.gz、train-labels-idx1-ubyte.gz、
t10k-images-idx3-ubyte.gz、t10k-labels-idx1-ubyte.gz 四个文件。

输出:mnist_mini.gz —— 每条记录 785 字节 = 1 字节标签 + 784 字节像素(0~255),
    先训练集后测试集,两类各 2500 条(每类按原始顺序取前若干条,数字均衡)。

用法:python gen_mnist_mini.py [IDX 文件所在目录] [输出文件路径]
    默认在当前目录找 IDX 文件,输出 mnist_mini.gz 到脚本所在目录。
"""
import collections
import gzip
import os
import struct
import sys


def read_idx(path):
    with gzip.open(path, "rb") as f:
        return f.read()


def parse_images(blob):
    magic, n, rows, cols = struct.unpack(">IIII", blob[:16])
    assert magic == 2051, "图片文件格式不对(魔数 %d)" % magic
    assert (rows, cols) == (28, 28)
    return n, blob[16:]


def parse_labels(blob):
    magic, n = struct.unpack(">II", blob[:8])
    assert magic == 2049, "标签文件格式不对(魔数 %d)" % magic
    return n, blob[8:]


def build_balanced(img_blob, lab_blob, total_n, per_digit):
    """按类别均衡抽取:每个数字取原始顺序的前 per_digit 条。"""
    buckets = {d: [] for d in range(10)}
    for i in range(total_n):
        buckets[lab_blob[i]].append(i)          # lab_blob[i] 是 int(py3)
    recs = []
    for d in range(10):
        for i in buckets[d][:per_digit]:
            off = i * 784
            recs.append(bytes([d]) + img_blob[off:off + 784])
    return recs


def render(rec):
    """把一条记录画成 ASCII,肉眼验收数据没读错。"""
    px = rec[1:]
    return "\n".join(
        "".join("#" if px[r * 28 + c] > 128 else ("+" if px[r * 28 + c] > 32 else ".")
                for c in range(28))
        for r in range(28))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "."
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mnist_mini.gz")

    n, timg = parse_images(read_idx(os.path.join(src, "train-images-idx3-ubyte.gz")))
    tn, tlab = parse_labels(read_idx(os.path.join(src, "train-labels-idx1-ubyte.gz")))
    m, simg = parse_images(read_idx(os.path.join(src, "t10k-images-idx3-ubyte.gz")))
    mm, slab = parse_labels(read_idx(os.path.join(src, "t10k-labels-idx1-ubyte.gz")))
    assert n == tn == 60000 and m == mm == 10000, "文件不完整,请重新下载"

    train_rec = build_balanced(timg, tlab, n, 200)   # 训练 200×10 = 2000
    test_rec = build_balanced(simg, slab, m, 50)     # 测试   50×10 =  500

    with gzip.open(dst, "wb", compresslevel=9) as f:
        f.write(b"".join(train_rec) + b"".join(test_rec))

    print("训练集每类张数:", dict(sorted(collections.Counter(r[0] for r in train_rec).items())))
    print("测试集每类张数:", dict(sorted(collections.Counter(r[0] for r in test_rec).items())))
    print("已写出:", dst, "(%d 字节)" % os.path.getsize(dst))
    print("train[0] 的标签 =", train_rec[0][0], "长这样:")
    print(render(train_rec[0]))


if __name__ == "__main__":
    main()
