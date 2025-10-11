import cv2
import typing
import numpy as np
import editdistance

from mltu.inferenceModel import OnnxInferenceModel
def ctc_decoder(preds, vocab):
    texts = []
    for pred in preds:
        seq = np.argmax(pred, axis=-1)
        prev_char = None
        text = ""
        for char in seq:
            if char != prev_char and char < len(vocab):
                text += vocab[char]
            prev_char = char
        texts.append(text)
    return texts

def get_cer(pred, gt):
    """Character Error Rate"""
    return editdistance.eval(pred, gt) / len(gt)

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        print("DEBUG — self.input_shape:", self.input_shape)

        input_shape = self.input_shape

        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
            h, w = input_shape[0:2]

        elif isinstance(input_shape, list) and len(input_shape) == 1 and isinstance(input_shape[0], (list, tuple)):
            h, w = input_shape[0][1:3]

        elif isinstance(input_shape, tuple) and len(input_shape) == 4:
            h, w = input_shape[1:3]

        else:
            print(f"⚠️ Tidak mengenali format input_shape {input_shape}, gunakan default (32,128)")
            h, w = 32, 128

        image = cv2.resize(image, (w, h))
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name

        preds = self.model.run([output_name], {input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("Models/03_handwriting_recognition/202510072024/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    df = pd.read_csv("Models/03_handwriting_recognition/202510072024/val.csv").values.tolist()

    accum_cer = []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

        accum_cer.append(cer)

        # resize by 4x
        image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Average CER: {np.average(accum_cer)}")