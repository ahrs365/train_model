import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

print(tf.__version__)

# 绘制训练和验证的准确率和损失图
def plot_training_history(history):
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
    plt.show()

# 评估和预测结果的可视化
def visualize_predictions(model, test_dataset, class_names):
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict(image_batch)
    predicted_class_indices = np.argmax(predictions, axis=1)

    print("Predictions:\n", predicted_class_indices)
    print("Labels:\n", label_batch)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].squeeze(), cmap="gray")  # 指定颜色映射为灰度
        plt.title(class_names[predicted_class_indices[i]])
        plt.axis("off")

    plt.show()

# 数据路径
data_dir = "data/tushuguan"

# 图像尺寸和批次大小
IMG_SIZE = (96, 96)
IMG_SHAPE = IMG_SIZE + (1,)
BATCH_SIZE = 32
base_learning_rate = 0.001
initial_epochs = 40

# 数据增强
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# 图像预处理函数
def preprocess_image(image, label):
    image = image / 255.0  # 归一化
    return image, label

# 创建训练和验证数据集
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="int",
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="int",
)

# 获取类名
class_names = train_dataset.class_names
print("Classes:", class_names)

# 应用图像预处理
train_dataset = train_dataset.map(preprocess_image)
validation_dataset = validation_dataset.map(preprocess_image)

# 数据增强应用到训练数据集
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# 使用 tf.data.experimental.cardinality 确定验证集中有多少批次的数据，然后将其中的 20% 移至测试集。
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 2)
validation_dataset = validation_dataset.skip(val_batches // 2)
print(
    "Number of validation batches: %d"
    % tf.data.experimental.cardinality(validation_dataset)
)
print("Number of test batches: %d" % tf.data.experimental.cardinality(test_dataset))

# 创建 MobileNet V1 模型
optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
input_tensor = tf.keras.Input(shape=IMG_SHAPE)
mobilenet_model = tf.keras.applications.MobileNet(
    input_shape=IMG_SHAPE,
    input_tensor=input_tensor,
    pooling="avg",
    alpha=0.25,
    weights=None,  # 我们将手动加载权重
    include_top=False
)
mobilenet_model.trainable = False

# 手动加载权重
mobilenet_model.load_weights("h5/mobilenetV1_0.25_96x96_greyscale_weights.h5")

mobilenet_output = mobilenet_model.output

# Dense layer
dense_layer = tf.keras.layers.Dense(256, activation="relu")(mobilenet_output)

# Dropout layer
dropout_layer = tf.keras.layers.Dropout(0.1)(dense_layer)

# Classification layer
classification_layer = tf.keras.layers.Dense(len(class_names), activation='softmax')(dropout_layer)

model = tf.keras.Model(inputs=mobilenet_model.input, outputs=classification_layer)

print("Compiling model...")
model.compile(loss="sparse_categorical_crossentropy",  # 使用 sparse_categorical_crossentropy 与整数标签
              optimizer=optimizer,
              metrics=["accuracy"])

model.summary()

# 缓存和预取数据
train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# 训练模型
history = model.fit(
    train_dataset, epochs=initial_epochs, validation_data=validation_dataset
)

# 评估模型
loss, accuracy = model.evaluate(test_dataset)
print("微调前 Test accuracy:", accuracy)

# 微调
mobilenet_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(mobilenet_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 50

# Freeze all the layers before the `fine_tune_at` layer
for layer in mobilenet_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate / 10),
    loss="sparse_categorical_crossentropy",
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)
model.summary()
len(model.trainable_variables)
fine_tune_epochs = 100
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset,
)
loss, accuracy = model.evaluate(test_dataset)
print("微调后 Test accuracy :", accuracy)

# 绘制训练历史
plot_training_history(history)
# 可视化测试集上的预测结果
visualize_predictions(model, test_dataset, class_names)

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

# Save the model.
with open("model/q_model.tflite", "wb") as f:
    f.write(tflite_model)

model_size = os.path.getsize("model/q_model.tflite")
print("q_model is %d bytes" % model_size)

# xxd -i model/model.tflite > model_data.cc
