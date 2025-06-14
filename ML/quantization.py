import tensorflow as tf

#Used for testing but this is the shape it should take
default_input = "80_80_files/UFDL/models/UFDL_no_sobel.h5"

def main(input = default_input):
    if isinstance(input, str):
        model = tf.keras.models.load_model(input, compile=False)
    else:
        model = input

    # Set up the converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Force full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # other optimizers have been rendered obscolete
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    #  Correct and real representative dataset
    def representative_dataset():
        for _ in range(100):
            dummy = tf.random.uniform([1, 40, 40, 1], minval=0, maxval=1, dtype=tf.float32)
            yield [dummy]

    converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    # save at temporary location to read it's size
    with open("temp.tflite", "wb") as f:
        f.write(tflite_model)

    print("Full int8 quantized model saved.")
