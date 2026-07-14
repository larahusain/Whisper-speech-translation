import argparse
import torch
import evaluate
import csv  # Added for TSV export
from datasets import load_dataset
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

def main(args):
    # Set up hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load ASR Model (Whisper Small - Baseline)
    print(f"Loading ASR model: {args.asr_model}")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=args.asr_model,
        device=0 if device == "cuda" else -1,
        generate_kwargs={"language": "chinese"}
    )

    # 2. Load MT Model (NLLB 600M)
    print(f"Loading MT model: {args.mt_model}")
    mt_tokenizer = AutoTokenizer.from_pretrained(args.mt_model, src_lang="yue_Hant")
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(args.mt_model).to(device)

    tgt_lang_id = mt_tokenizer.convert_tokens_to_ids("eng_Latn")

    # 3. Load Dataset
    print(f"Loading FLEURS dataset for Cantonese (yue_hant_hk), split: {args.split}")
    dataset = load_dataset("google/fleurs", "yue_hant_hk", split=args.split)

    if args.limit:
        dataset = dataset.select(range(args.limit))
        print(f"Limiting evaluation to {args.limit} samples.")

    # 4. Load Metrics
    wer_metric = evaluate.load("wer")
    bleu_metric = evaluate.load("sacrebleu")

    predictions_asr = []
    references_asr = []
    predictions_mt = []
    references_mt = []

    # Initialize list to store results for TSV
    all_results = []

    print("Starting evaluation pipeline...")
    for i, item in enumerate(dataset):
        ref_audio_text = item["raw_transcription"]
        ref_english_text = item["transcription"]

        # --- ASR Step ---
        audio_input = item["audio"]["array"]
        asr_result = asr_pipeline(audio_input)
        transcribed_text = asr_result["text"]

        # --- MT Step ---
        inputs = mt_tokenizer(transcribed_text, return_tensors="pt").to(device)
        translated_tokens = mt_model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=400
        )
        translated_text = mt_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        # Store for metrics
        predictions_asr.append(transcribed_text)
        references_asr.append(ref_audio_text)
        predictions_mt.append(translated_text)
        references_mt.append(ref_english_text)

        # Collect for TSV
        all_results.append({
            "1.Reference_Cantonese_Text": ref_audio_text,
            "2.Whisper_ASR_Prediction": transcribed_text,
            "3.Reference_English_Target": ref_english_text,
            "4.MT_English_Prediction": translated_text
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} samples...")

    # 5. Compute Metrics
    wer_score = wer_metric.compute(predictions=predictions_asr, references=references_asr)
    bleu_score = bleu_metric.compute(predictions=predictions_mt, references=[[ref] for ref in references_mt])

    # 6. Output Results and Save TSV
    print("\n" + "="*40)
    print("CANTONESE EVALUATION RESULTS")
    print("="*40)
    print(f"ASR Word Error Rate (WER):  {wer_score:.4f}")
    print(f"MT Translation Score (BLEU): {bleu_score['score']:.2f}")
    print("="*40)

    with open('pipeline_results_yue_detailed.tsv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "1.Reference_Cantonese_Text",
            "2.Whisper_ASR_Prediction",
            "3.Reference_English_Target",
            "4.MT_English_Prediction"
        ], delimiter='\t')
        writer.writeheader()
        writer.writerows(all_results)
    print("Detailed results saved to pipeline_results_yue_detailed.tsv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cantonese ASR to MT Pipeline")
    parser.add_argument("--asr_model", type=str, default="openai/whisper-small")
    parser.add_argument("--mt_model", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args)