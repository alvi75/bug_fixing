import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import sys
import json
from codebleu import calc_codebleu
from train import Vocab, BugDataset, Model, collate_fn
import os

BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model_path, model_type='lstm'):

    print(f"evaluation {model_type.upper()} model")
    print(f"Model path: {model_path}")
    print(f"Device: {DEVICE}")
    print("-" * 50)

    print("\nLoading test data....")
    dataset = load_dataset("google/code_x_glue_cc_code_refinement", name="small")

    train_data = list(dataset['train'])
    test_data = list(dataset['test'])

    print(f"test samples: {len(test_data)}")

    print("\nBuilding vocabulary...")
    vocab = Vocab()
    vocab.build(train_data)
    print(f"Vocab size: {len(vocab.token2idx)}")

    test_dataset = BugDataset(test_data, vocab)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("\nloading model...")
    model = Model(428, model_type).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    print("model loaded successfully...")

    print("\ngenerating predictions for test set...")
    predictions = []
    references = []

    with torch.no_grad():
        for src, tgt in tqdm(test_loader, desc="Evaluating"):
            src, tft = src.to(DEVICE), tgt.to(DEVICE)

            for i in range(src.size(0)):
                pred_ids = model.generate(src[i:i+1])
                pred_text = vocab.decode(pred_ids)

                ref_text = vocab.decode(tgt[i].tolist())

                predictions.append(pred_text)
                references.append([ref_text])

    print(f"\nGenerated {len(predictions)} predictions")

    save_dir = "predictions"
    os.makedirs(save_dir, exist_ok=True)
    output_file = f"{save_dir}/{model_type}_predictions.json"
    print(f"\saving predictions to {output_file}...")

    save_data = {
        "model_type": model_type,
        "model_path": model_path,
        "test_samples": len(predictions),
        "predictions": []
    }

    for i in range(len(predictions)):
        save_data["predictions"].append({
            "id": i,
            "reference": references[i][0],
            "predicted": predictions[i],
            "exact_match": predictions[i] == references[i][0]
        })

    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"Predictions saved to {output_file}")

    print("\nCalculating CodeBLEU score...")
    result = calc_codebleu(references, predictions, lang='java')

    print("\n" + "-"*50)
    print("evaluation results")
    print(f"Model: {model_type.upper()}")
    print(f"test samples: {len(predictions)}")
    print(f"CodeBLEU score: {result['codebleu']: .4f}")
    print(f"output file: {output_file}")

    return result['codebleu']
    
def main():
    if len(sys.argv) < 2:
        print("Usage: python eval.py model_path lstm or gru")
        print("example: python eval.py lstm_small.pt lstm")
        return
    
    model_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'lstm'

    if not os.path.exists(model_path):
        print(f"Error: model file '{model_path}' not found")
        return
    
    score = evaluate_model(model_path, model_type)

    print(f"final CodeBLEU score: {score: .4f} ({score*100: .2f}%)")

if __name__ == "__main__":
    main()