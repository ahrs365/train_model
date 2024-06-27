import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# 检查TensorFlow版本和可用的GPU
print(tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("可用的 GPU：")
    for gpu in gpus:
        print(gpu)
else:
    print("未检测到 GPU。")

# 数据路径
data_dir = "data/arm"

# 图像尺寸和批次大小
IMG_SIZE = (96, 96)
BATCH_SIZE = 16

# 创建训练和验证数据集
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
)

# 使用 tf.data.experimental.cardinality 确定验证集中有多少批次的数据，然后将其中的 20% 移至测试集。
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 2)
validation_dataset = validation_dataset.skip(val_batches // 2)

# 获取类名
class_names = train_dataset.class_names
print("Classes:", class_names)

# 数据增强
data_augmentation = tf.keras.Sequential(
    [tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.2)]
)

# 构建更小的CNN模型
model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(96, 96, 1)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(8, (3, 3), activation="relu"),  # 减少过滤器数量
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),  # 减少过滤器数量
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),  # 减少全连接层神经元数量
        tf.keras.layers.Dropout(0.5),  # 保留 Dropout 层
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ]
)


# 编译模型
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# 模型摘要
model.summary()

# 缓存和预取数据
train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# 训练模型
initial_epochs = 40
history = model.fit(
    train_dataset, epochs=initial_epochs, validation_data=validation_dataset
)

# 绘制训练和验证的准确率和损失图
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, max(plt.ylim())])
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()


# 评估和预测
loss, accuracy = model.evaluate(test_dataset)
print("Test accuracy :", accuracy)

# Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict(image_batch)
predicted_class_indices = np.argmax(predictions, axis=1)

print("Predictions:\n", predicted_class_indices)
print("Labels:\n", label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predicted_class_indices[i]])
    plt.axis("off")

plt.show()


# 保存模型为TensorFlow Lite模型
def representative_data_gen():
    for images, _ in train_dataset.take(100):
        # images.numpy() 转换图像张量到 numpy 数组，适用于 TensorFlow Lite 预处理
        yield [images.numpy()]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

test_gen = representative_data_gen()
sample_input = next(test_gen)
print("Sample input shape:", sample_input[0].shape)


# 确保模型转换后只使用整数
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

# 保存模型
with open("model/gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

print("模型已保存为gesture_model.tflite")
model_size = os.path.getsize("model/gesture_model.tflite")
print("gesture_model is %d bytes" % model_size)


# xxd -i model/gesture_model.tflite > gesture_data.cc
