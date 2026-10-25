# -*- coding: utf-8 -*-
"""
基于卷积神经网络的交通标志图像分类识别
运行环境：TensorFlow 2.21.0 + Python 3.12（CPU：Intel i5-8365U，4核8线程，16GB 内存）

本机没有独立显卡（Intel UHD 620 核显不参与 TensorFlow 计算），训练完全依赖 CPU，
因此代码针对 CPU 做了如下优化：
1. 启用 oneDNN（oneAPI 深度神经网络库），利用 CPU 的 AVX2/FMA 指令集加速卷积运算；
2. 按物理核心数限制 intra-op 线程、并限制 inter-op 线程，避免 4 核 8 线程过度订阅、线程争抢；
3. 图像只解码一次并缓存进内存（4200 张 32×32 图片仅几十 MB），避免每个 epoch 重复读盘、解码；
4. 用 shuffle + prefetch 流水线保证每个 epoch 真正打乱样本，并让数据准备与计算重叠；
5. 适当增大 batch_size（16→32），提高 CPU 上矩阵乘法的吞吐效率。

注意：验证集必须与训练集使用相同的 shuffle 和 seed，二者才是互补划分；若验证集用
shuffle=False，会按类别顺序切分，导致验证集/测试集只包含部分类别，评估结果失真。
"""
import os
# 启用 oneDNN CPU 加速，并减少 TensorFlow 冗余日志（必须在导入 tensorflow 之前设置）
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# CPU 线程配置通过环境变量完成：intra-op 取物理核心数（本机 8 逻辑线程 → 4 个物理核心），
# inter-op 取 2，兼顾并行算子与散热，避免笔记本 CPU 因线程争抢而降频。
# 注意：不能直接调用 tf.config.threading.set_...，因为在 Spyder 中重复运行时
# TensorFlow 运行时已初始化，会抛出 "Intra op parallelism cannot be modified after initialization"。
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', str(max(1, (os.cpu_count() or 4) // 2)))
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 固定随机种子，保证实验可复现
tf.keras.utils.set_random_seed(123)

# 设置中文字体，解决图表中文显示为方框的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#  1. 数据集加载与自动划分
# 数据集根目录路径
data_root = r"G:\深度学习与开发\Jupyter\traffic\train"
img_shape = (32, 32)  # 统一输入图像尺寸
batch_size = 32       # 训练批次大小（CPU 上适当增大批次可提高吞吐）
AUTOTUNE = tf.data.AUTOTUNE

# 第一步：从总数据中拆分70%训练集 + 30%临时验证集
# 两个子集都设置 shuffle=True 且 seed 相同，保证划分互补，且各类别分布均匀
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_root,
    validation_split=0.3,
    subset="training",
    seed=123,          # 固定随机种子，保证实验可复现
    image_size=img_shape,
    batch_size=None,   # 先按单样本读取，便于整体缓存和打乱
    shuffle=True
)

val_temp_ds = tf.keras.utils.image_dataset_from_directory(
    data_root,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=img_shape,
    batch_size=None,
    shuffle=True
)

# 第二步：从临时验证集中再拆分，一半做验证集、一半做测试集
val_temp_ds = val_temp_ds.cache()          # 验证/测试图片同样只解码一次
val_size = int(len(val_temp_ds) * 0.5)
val_ds = val_temp_ds.take(val_size)        # 验证集
test_ds = val_temp_ds.skip(val_size)       # 测试集

# 先保存类别名称（必须在map操作之前执行，map后丢失元数据属性）
class_names = train_ds.class_names
print("\n数据集类别：", class_names)

# 像素归一化处理：将0~255的像素值映射到0~1区间，加速收敛
norm_layer = tf.keras.layers.Rescaling(1./255)

# 数据流水线优化：cache 把解码后的图片常驻内存，之后每个 epoch 无需再读盘；
# shuffle 打乱全部训练样本；prefetch 让 CPU 在训练当前批次时准备下一批数据
train_count = len(train_ds)
train_ds = (train_ds
            .cache()
            .shuffle(train_count, seed=123)
            .map(lambda x, y: (norm_layer(x), y))
            .batch(batch_size)
            .prefetch(AUTOTUNE))
val_ds = val_ds.map(lambda x, y: (norm_layer(x), y)).batch(batch_size).prefetch(AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (norm_layer(x), y)).batch(batch_size).prefetch(AUTOTUNE)

# 2. 数据集样本可视化（图1）
def show_sample_images():
    """随机抽取训练集样本并可视化，验证数据读取正确性"""
    plt.figure(figsize=(10, 4))
    # 取一个批次的样本
    for images, labels in train_ds.take(1):
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(images[i].numpy())
            plt.title(f"类别{class_names[labels[i]]}")
            plt.axis("off")
    plt.suptitle("图1 交通标志数据集样本展示")
    plt.tight_layout()
    plt.show()

# 执行样本可视化
show_sample_images()

# 3. 构建10层卷积神经网络
model = tf.keras.Sequential([
    # 第1层：卷积层，32个3×3卷积核，提取底层边缘、颜色特征
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)),
    # 第2层：卷积层，强化底层特征的融合与抽象
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    # 第3层：最大池化层，降维压缩特征图，减少计算量
    tf.keras.layers.MaxPooling2D((2,2)),
    # 第4层：卷积层，64个3×3卷积核，提取中层纹理、形状特征
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    # 第5层：卷积层，深化中层语义特征表达
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    # 第6层：最大池化层，二次降维，保留关键特征
    tf.keras.layers.MaxPooling2D((2,2)),
    # 第7层：展平层，将二维特征图转换为一维向量
    tf.keras.layers.Flatten(),
    # 第8层：全连接层，整合全局特征进行分类推理
    tf.keras.layers.Dense(128, activation='relu'),
    # 第9层：Dropout层，随机失活30%神经元，缓解过拟合
    tf.keras.layers.Dropout(0.3),
    # 第10层：输出层，5分类，softmax输出类别概率分布
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# 打印网络结构详情，可截图放入实验报告
print("\n===== 卷积神经网络结构 =====")
model.summary()

# 模型编译：配置优化器、损失函数、评估指标
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. 模型训练
print("\n开始模型训练...")
start_time = time.perf_counter()   # 记录训练耗时，便于对比优化前后的性能
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20  # 总训练轮次
)
print(f"\n训练总耗时：{time.perf_counter() - start_time:.1f} 秒")

# 5. 训练曲线可视化（图2、图3）
plt.figure(figsize=(12, 5))
# 图2：准确率变化曲线
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='训练准确率', linewidth=2)
plt.plot(history.history['val_accuracy'], label='验证准确率', linewidth=2, linestyle='--')
plt.title('图2 模型准确率变化曲线')
plt.xlabel('迭代轮次 Epoch')
plt.ylabel('准确率 Accuracy')
plt.legend()
plt.grid(alpha=0.3)

# 图3：损失变化曲线
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='训练损失', linewidth=2)
plt.plot(history.history['val_loss'], label='验证损失', linewidth=2, linestyle='--')
plt.title('图3 模型损失变化曲线')
plt.xlabel('迭代轮次 Epoch')
plt.ylabel('损失值 Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#  6. 测试集评估与单样本预测（图4）
# 测试集整体性能评估
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\n===== 测试集最终性能 =====")
print(f"测试集损失：{test_loss:.4f}")
print(f"测试集准确率：{test_acc:.2%}")

# 单样本预测演示
def predict_test_sample():
    """从测试集取单张图片进行预测并可视化结果"""
    for images, labels in test_ds.take(1):
        img = images[0]
        true_label = class_names[labels[0]]
        # 模型预测，扩展batch维度
        pred = model.predict(tf.expand_dims(img, 0), verbose=0)
        pred_label = class_names[np.argmax(pred)]
        confidence = np.max(pred)

        # 绘制预测结果
        plt.figure()
        plt.imshow(img.numpy())
        plt.title(f'图4 单样本预测结果\n真实类别：{true_label}\n预测类别：{pred_label}  置信度：{confidence:.2%}')
        plt.axis('off')
        plt.show()

# 执行预测演示
predict_test_sample()
