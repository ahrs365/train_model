import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

print(tf.__version__)

# 列出所有可用的设备，并检查是否有 GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("可用的 GPU：")
    for gpu in gpus:
        print(gpu)
else:
    print("未检测到 GPU。")
    
# 数据路径
data_dir = "data/rps_data_sample"

# 图像尺寸和批次大小
IMG_SIZE = (160, 160)
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
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, include_top=False, alpha=0.35, weights="imagenet"
)
base_model.trainable = False  # 冻结模型

# 数据预处理
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# 构建模型
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(class_names), activation="softmax")

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)  # 应用数据增强
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
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
initial_epochs = 10
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

# 微调
base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate / 10),
    loss="sparse_categorical_crossentropy",
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)
model.summary()
len(model.trainable_variables)
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset,
)

acc.extend(history_fine.history["accuracy"])
val_acc.extend(history_fine.history["val_accuracy"])
loss.extend(history_fine.history["loss"])
val_loss.extend(history_fine.history["val_loss"])

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.ylim([0.8, 1])
plt.plot(
    [initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning"
)
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.ylim([0, 1.0])
plt.plot(
    [initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning"
)
plt.legend(loc="upper right")
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

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
q_tflite_model = converter.convert()

# Save the model.
with open("model/q_model.tflite", "wb") as f:
    f.write(q_tflite_model)

model_size = os.path.getsize("model/model.tflite")
print("model is %d bytes" % model_size)

model_size = os.path.getsize("model/q_model.tflite")
print("q_model is %d bytes" % model_size)

# xxd -i model/model.tflite > model_data.cc
