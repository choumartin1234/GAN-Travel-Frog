import tensorflow as tf

saved_model_dir='model'
converter=tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model=converter.convert()
open('model.tflite','wb').write(tflite_model)