import argparse
import torch
import pandas as pd
from datasets import load_dataset
from transformers import pipeline, MarianMTModel, MarianTokenizer
from evaluate import load

def is_valid_audio(example):
    try:
        _ = example["audio"]["array"]
        return True
    except Exception:
        return False

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    print(f"Loading ASR model from: {args.asr_model}")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=args.asr_model,
        device=device,
        generate_kwargs={"language": args.asr_language, "task": "transcribe"}
    )

    print(f"Loading MT model: {args.mt_model}")
    # Bypassing the buggy pipeline and loading the model directly
    mt_tokenizer = MarianTokenizer.from_pretrained(args.mt_model)
    mt_model = MarianMTModel.from_pretrained(args.mt_model).to(device)


    wer_metric = load("wer")
    bleu_metric = load("sacrebleu")


    print(f"Loading FLEURS dataset for language: {args.dataset_lang}")
    eval_dataset = load_dataset("google/fleurs", args.dataset_lang, split="test")
    fleurs_en_test = load_dataset("google/fleurs", "en_us", split="test")
    en_translations = {item["id"]: item["raw_transcription"] for item in fleurs_en_test}


    asr_predictions = []
    asr_references = []
    mt_predictions = []
    mt_references = []

    dataset_to_eval = eval_dataset.select(range(args.limit)) if args.limit else eval_dataset
    total_samples = len(dataset_to_eval)

    print(f"Starting evaluation on {total_samples} samples...")

    for i, item in enumerate(dataset_to_eval):
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total_samples} samples...")

        try:
            audio = item["audio"]["array"]
            ref_source_text = item["raw_transcription"]
            ref_english_text = en_translations[item["id"]]

            # Step A: ASR (Audio -> Source Text)
            asr_result = asr_pipeline(audio)["text"]
            asr_predictions.append(asr_result)
            asr_references.append(ref_source_text)

            # Step B: MT (Source Text -> English Text)
            inputs = mt_tokenizer(asr_result, return_tensors="pt", padding=True).to(device)
            translated_tokens = mt_model.generate(**inputs)
            mt_result = mt_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

            mt_predictions.append(mt_result)
            mt_references.append([ref_english_text])

        except Exception as e:
            print(f"Skipping sample {i+1} due to corrupted audio file")
            continue

    # Calculating final metrics
    wer_score = wer_metric.compute(predictions=asr_predictions, references=asr_references)
    bleu_score = bleu_metric.compute(predictions=mt_predictions, references=mt_references)

    print("Results")
    print(f"Language Evaluated : {args.dataset_lang}")
    print(f"ASR WER            : {wer_score:.4f}")
    print(f"Final MT BLEU      : {bleu_score['score']:.2f}")

    # Saving results into file
    flat_mt_references = [ref[0] for ref in mt_references]
    results_df = pd.DataFrame({
        "1.Reference_Arabic_Audio": asr_references,
        "2.Whisper_ASR_Prediction": asr_predictions,
        "3.Reference_English_Target": flat_mt_references,
        "4.MT_English_Prediction": mt_predictions
    })

    output_filename = f"pipeline_results_{args.dataset_lang}.tsv"
    results_df.to_csv(output_filename, sep="\t", index=False, encoding="utf-8")
    print("Completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Cascaded ASR -> MT Pipeline")
    parser.add_argument("--asr_model", type=str, default="./whisper-small-arabic-asr-final")
    parser.add_argument("--mt_model", type=str, default="Helsinki-NLP/opus-mt-ar-en")
    parser.add_argument("--dataset_lang", type=str, default="ar_eg")
    parser.add_argument("--asr_language", type=str, default="arabic")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args)