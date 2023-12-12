import tensorflow as tf

base_model = tf.keras.applications.MobileNetV2(weights='imagenet')

base_model.save('mobilenetv2.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
tflite_model = converter.convert()

with open('mobilenetv2_1.00_224_quant.tflite', 'wb') as f:
    f.write(tflite_model)
