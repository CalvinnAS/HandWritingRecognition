import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class TrOCRHandwritingRecognizer:
    def __init__(self, model_name="microsoft/trocr-base-handwritten", device="cpu"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)

    def predict(self, image: np.ndarray) -> str:
        """Recognize handwritten text from a numpy image."""
        if image is None or not isinstance(image, np.ndarray):
            return ""

        if len(image.shape) == 2:  # grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:  # single channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_pil = Image.fromarray(image)

        pixel_values = self.processor(
            images=image_pil,
            return_tensors="pt",
            input_data_format="channels_last"
        ).pixel_values.to(self.device)

        generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return text.strip()


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv("Models/03_handwriting_recognition/202510072024/val.csv").values.tolist()

    recognizer = TrOCRHandwritingRecognizer()

    for image_path, label in tqdm(df[:10]):
        image = cv2.imread(image_path)
        prediction = recognizer.predict(image)
        print(f"GT: {label} | Pred: {prediction}")

if __name__ == "__main__":
    from PIL import Image

    import cv2
    import os

    image_path = "test.png"
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        text = recognizer.predict(image)
        print(f"\nPrediction '{image_path}': {text}")
    else:
        print("Image not found.")


