from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import torch
from PIL import Image
import os
from datasets import disable_caching

disable_caching()

BASE_DIR = "Datasets_TrOCR/IAM_Words/"
dataset_path = "Datasets_TrOCR/IAM_Words/dataset_TrOCR.csv"
model_name = "microsoft/trocr-base-handwritten"

processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

def is_image_valid(example):
    try:
        img_path = example['image_path']
        img_path = img_path.replace("\\", "/")
        full_path = os.path.join(BASE_DIR, img_path)
        full_path = full_path.replace("\\", "/")

        if not os.path.exists(full_path):
            return False

        with Image.open(full_path) as img:
            img.verify()

            with Image.open(full_path) as valid_img:
                if valid_img.size[0] < 5 or valid_img.size[1] < 5:
                    return False

        return True
    except Exception:
        return False

def transform(batch):

    images = []
    texts = []

    for idx, img_path in enumerate(batch["image_path"]):
        try:
            img_path = img_path.replace("\\", "/")
            full_path = os.path.join(BASE_DIR, img_path)
            full_path = full_path.replace("\\", "/")

            image = Image.open(full_path).convert("RGB")

            images.append(image)
            texts.append(batch["text"][idx])

        except Exception as e:
            print(f"Error loading {img_path} during training: {e}")
            images.append(Image.new('RGB', (384, 384), color='black'))
            texts.append("")

    pixel_values = processor(images=images, return_tensors="pt").pixel_values

    labels = processor.tokenizer(
        texts,
        padding="max_length",
        truncation=True
    ).input_ids

    labels = [
        [(l if l != processor.tokenizer.pad_token_id else -100) for l in label]
        for label in labels
    ]

    return {"pixel_values": pixel_values, "labels": labels}



print("Loading dataset...")
dataset = load_dataset("csv", data_files=dataset_path)
dataset = dataset["train"]

print(f"Jumlah data awal: {len(dataset)}")
print("Memfilter gambar rusak/kecil...")
dataset = dataset.filter(is_image_valid)
print(f"Jumlah data valid: {len(dataset)}")

dataset.set_transform(transform)

training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr_finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    eval_strategy="no",
    fp16=torch.cuda.is_available(),
    learning_rate=5e-5,
    num_train_epochs=10,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_num_workers=0
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,
)


print("Mulai Training...")
trainer.train()


model.save_pretrained("./trocr_finetuned")
processor.save_pretrained("./trocr_finetuned")
print("Training Selesai & Model Disimpan.")