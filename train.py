import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
import sys
from codebleu import calc_codebleu
import os

BATCH_SIZE = 32
MAX_LEN = 100
EVAL_STEPS = 500
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Vocab:
    def __init__(self):
        self.token2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, "<UNK>": 3}
        self.idx2token = {}

    def build(self, data):
        counter = Counter()
        for sample in data:
            counter.update(sample['buggy'].split() + sample['fixed'].split())

        idx = 4
        for token, count in counter.items():
            if count >= 2:
                self.token2idx[token] = idx
                idx += 1

        self.idx2token = {v: k for k, v in self.token2idx.items()}

    def encode(self, text):
        tokens = [1] + [self.token2idx.get(t, 3) for t in text.split()[:98]] + [2]
        return tokens 

    def decode(self, indices):
        return ' '.join([self.idx2token.get(i, '') for i in indices if i not in [0,1,2]])
    
class BugDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.vocab.encode(self.data[idx]['buggy'])),
                torch.tensor(self.vocab.encode(self.data[idx]['fixed'])))
    
def collate_fn(batch):
    buggy, fixed = zip(*batch)
    return (pad_sequence(buggy, batch_first=True),
            pad_sequence(fixed, batch_first=True))

class Model(nn.Module):
    def __init__(self, vocab_size, rnn_type = 'lstm'):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 256, padding_idx=0)
        # self.dropout = nn.Dropout(0.2) 

        if rnn_type == 'lstm':
            self.encoder = nn.LSTM(256, 256, 2, batch_first=True)
            self.decoder = nn.LSTM(256, 256, 2, batch_first=True)
        else:
            self.encoder = nn.GRU(256, 256, 2, batch_first=True)
            self.decoder = nn.GRU(256, 256, 2, batch_first=True)

        self.out = nn.Linear(256, vocab_size)

    def forward(self, src, tgt):
        _, hidden = self.encoder(self.embed(src))
        output, _ = self.decoder(self.embed(tgt), hidden)
        return self.out(output)
    
    def generate(self, src):
        _, hidden = self.encoder(self.embed(src))
        outputs = []
        input_tok = torch.tensor([[1]]).to(src.device)

        for _ in range(MAX_LEN):
            out, hidden = self.decoder(self.embed(input_tok), hidden)
            pred = self.out(out).argmax(-1)
            if pred.item() == 2:
                break
            outputs.append(pred.item())
            input_tok = pred

        return outputs

def compute_codebleu(model, val_loader, vocab):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        count = 0
        for src, tgt in val_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)

            for i in range(src.size(0)):
                if count >= 100:
                    break
            
                pred_ids = model.generate(src[i:i+1])
                pred_text = vocab.decode(pred_ids)
                ref_text = vocab.decode(tgt[i].tolist())

                predictions.append(pred_text)
                references.append([ref_text])
                count += 1

            if count >= 100:
                break

    result = calc_codebleu(references, predictions, lang='java')
    return result['codebleu']


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['lstm', 'gru']:
        print("usage: python train.py lstm or gru")
        return
    
    model_type = sys.argv[1]
    print(f"Training {model_type.upper()} on small dataset")

    print("loading dataset (small)")
    dataset = load_dataset("google/code_x_glue_cc_code_refinement", name="small")

    train_data = list(dataset['train'])
    val_data = list(dataset['validation'])

    print("dataset size:")
    print(f"Train: {len(train_data)}")
    print(f"Val: {len(val_data)}")

    vocab = Vocab()
    vocab.build(train_data)
    print(f"Vocab size: {len(vocab.token2idx)}")

    train_loader = DataLoader(BugDataset(train_data, vocab), BATCH_SIZE, True, collate_fn=collate_fn)
    val_loader = DataLoader(BugDataset(val_data, vocab), BATCH_SIZE, False, collate_fn=collate_fn)

    steps_per_epoch = len(train_loader)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Will evaluate every {EVAL_STEPS} steps {steps_per_epoch/EVAL_STEPS: .1f} times per epoch")

    model = Model(len(vocab.token2idx), model_type).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_codebleu = 0
    patience = 0
    step = 0

    print("\nstarting training....")

    for epoch in range(10):
        model.train()
        epoch_loss = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for src, tgt in progress:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)

            output = model(src, tgt)
            loss = criterion(output[:, 1:].reshape(-1, output.size(-1)),
                             tgt[:, 1:].reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

            progress.set_postfix({'loss': f'{loss.item():.3f}'})

            if step % EVAL_STEPS == 0:
                codebleu = compute_codebleu(model, val_loader, vocab)

                print(f"\n Step {step}: CodeBLEU = {codebleu:.4f}", end="")

                if codebleu > best_codebleu:
                    best_codebleu = codebleu
                    patience = 0
                    save_dir = "checkpoints"
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(model.state_dict(), f'{save_dir}/{model_type}_small.pt')
                    print("Saved")

                else:
                    patience += 1
                    print(f"Patience {patience}/3")
                    if patience >= 3:
                        print(f"\nEarly stopping! Best CodeBLEU: {best_codebleu: .4f}")
                        return
                    
                model.train()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss: .3f}")

    print(f"\nTraining finished! Best CodeBLEU: {best_codebleu: .4f}")

if __name__ == "__main__":
    main()


