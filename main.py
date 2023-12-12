from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# TensorFlow Lite 모델 로드
model_path = "mobilenetv2_1.00_224_quant.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 입력 및 출력 텐서의 인덱스 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ImageNet 클래스 레이블 로드
with open('imagenet_labels.txt', 'r') as file:
    labels = file.read().splitlines()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 업로드한 이미지 처리
        file = request.files["file"]
        if file:
            img = Image.open(BytesIO(file.read()))
            img = img.resize((224, 224))
            input_frame = np.expand_dims(np.array(img), axis=0)
            input_frame = (input_frame / 255.0).astype(np.float32)

            # 모델 실행
            interpreter.set_tensor(input_details[0]['index'], input_frame)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # 결과 처리
            predicted_class = np.argmax(output_data[0])
            predicted_label = labels[predicted_class]

            return render_template("index.html", prediction=f"Prediction: {predicted_label} (Class {predicted_class})")

    return render_template("index.html", prediction="")


if __name__ == "__main__":
    app.run(debug=True)