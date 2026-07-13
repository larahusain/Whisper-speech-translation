# 1. dev_az_prediction300.tsv and dev_yue_prediction300.tsv 
# are generated using an original whisper-small model ("pred_before" column)
# and fine-tuned whisper-small-az and whisper-small-yue ("pred_after" column).

# 2. Except for "pred_before" and "pred_after", 
# other columns in the files are from original fleur dataset.

# 3. Cantonese model whisper-small-yue is trained using language code chinese.
# Azerbaijani model  whisper-small-az is trained using language code turkish 
# (this might not be the best option since whisper have an "azerbaijani" language code as well).

# 4. Both fine-tuned models were trained with the following parameters.
# Both model trainings stopped at steps=300, due to limit of GPU resource.

training_args = Seq2SeqTrainingArguments(
  
    output_dir="/content/drive/MyDrive/Whisper_FT/whisper-small-az",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=100,
    num_train_epochs=3,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    push_to_hub=False,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics_bleu, 
    processing_class=processor.feature_extractor,
)
