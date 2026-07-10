import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass
from typing import Any, Dict, List, Union


model_id = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_id, language="Arabic", task="transcribe")


def is_valid_audio(example):
    try:
        _ = example["audio"]["array"]
        return True
    except Exception:
        print("Removing corrupted audio file")
        return False

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["raw_transcription"]).input_ids
    return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def main():
    # Loading dataset
    train_dataset = load_dataset("google/fleurs", "ar_eg", split="train")
    test_dataset = load_dataset("google/fleurs", "ar_eg", split="test")

    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Getting rid of any corrupted data files
    train_dataset = train_dataset.filter(is_valid_audio, num_proc=2)
    test_dataset = test_dataset.filter(is_valid_audio, num_proc=2)

    print(f"Clean training samples ready: {len(train_dataset)}")

    # Extracting features
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=2)
    test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names, num_proc=2)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-arabic-asr",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=500,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100,
        eval_steps=100,
        logging_steps=25,
        report_to=["none"],
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    # Fine tuning
    trainer.train()

    print("Saved as final model")
    trainer.save_model("./whisper-small-arabic-asr-final")
    processor.save_pretrained("./whisper-small-arabic-asr-final")
    print("Completed")

if __name__ == '__main__':
    main()