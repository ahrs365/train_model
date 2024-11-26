# train_model
train a mobilbenet for classify
提供了4种训练脚本,用于手势/动作识别的训练：
1.直接搭建cnn
2.加载预训练好的H5模型
3.通过tf_hub加载模型
4.通过keras加载模型

注意确保模型转换后只使用整数，esp32只能运行整数模型
```
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
q_tflite_model = converter.convert()
```
