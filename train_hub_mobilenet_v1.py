import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import tensorflow_hub as hub


##############################################
#  加载hub上的预训练模型，进行训练                #
#  depth multiplier 0.35                     #
##############################################

print(tf.__version__)

# 列出所有可用的设备，并检查是否有 GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("可用的 GPU：")
    for gpu in gpus:
        print(gpu)
else:
    print("未检测到 GPU。")

# 数据路径
# data_dir = "data/rps_data_sample"
data_dir = "data/hagrid"


# 图像尺寸和批次大小
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

# 创建训练和验证数据集
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

# 使用 tf.data.experimental.cardinality 确定验证集中有多少批次的数据，然后将其中的 20% 移至测试集。
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 2)
validation_dataset = validation_dataset.skip(val_batches // 2)
print(
    "Number of validation batches: %d"
    % tf.data.experimental.cardinality(validation_dataset)
)
print("Number of test batches: %d" % tf.data.experimental.cardinality(test_dataset))

# 获取类名
class_names = train_dataset.class_names
print("Classes:", class_names)

# 数据增强
data_augmentation = tf.keras.Sequential(
    [tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.2)]
)


# 创建 MobileNet V2 模型
IMG_SHAPE = IMG_SIZE + (3,)
# base_model = tf.keras.applications.MobileNetV2(
#     input_shape=IMG_SHAPE, include_top=False, alpha=0.35, weights="imagenet"
# )
# base_model.trainable = False  # 冻结模型

base_model = hub.KerasLayer(
    "https://www.kaggle.com/models/google/mobilenet-v1/TensorFlow2/025-128-feature-vector/2",
    trainable=True,
    arguments=dict(batch_norm_momentum=0.997),
)


###################方式1：sequential####################
# model = tf.keras.Sequential([
#     base_model,
#     tf.keras.layers.Dense(len(class_names), activation='softmax')
# ])
# model.build([None, 128, 128, 3])  # Batch input shape.


####################方式2：函数api######################
# 数据预处理
preprocess_input = tf.keras.applications.mobilenet.preprocess_input

# 构建模型
# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(class_names), activation="softmax")

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)  # 应用数据增强
x = preprocess_input(x)
x = base_model(x)
# x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


# 编译模型
base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# 模型摘要
model.summary()

# 缓存和预取数据
train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# 训练模型
initial_epochs = 20
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
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 1.0])
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
# plt.show()


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


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open("model/model.tflite", "wb") as f:
    f.write(tflite_model)


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
q_tflite_model = converter.convert()

# Save the model.
with open("model/q_model.tflite", "wb") as f:
    f.write(q_tflite_model)

model_size = os.path.getsize("model/model.tflite")
print("model is %d bytes" % model_size)

model_size = os.path.getsize("model/q_model.tflite")
print("q_model is %d bytes" % model_size)

# xxd -i model/model.tflite > model_data.cc
