"""
Neural AI - Version PyTorch avec acceleration GPU automatique.
"""

import os
import pickle
import random
import re
import sys
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn


class Tokenizer:
    """Tokenizer simple avec vocabulaire persistant."""

    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.word_freq = Counter()
        self.vocab_size = 4

    def _tokenize(self, text):
        text = text.lower().strip()
        text = re.sub(r"[^\w\s']", " ", text)
        return text.split()

    def fit(self, texts):
        print("[TOKENIZER] Construction du vocabulaire...")
        self.word_freq = Counter()

        for text in texts:
            self.word_freq.update(self._tokenize(text))

        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.vocab_size = 4

        for word, _ in self.word_freq.most_common(self.max_vocab_size - 4):
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

        print(f"   - Vocabulaire: {self.vocab_size} mots")
        print(f"   - Mots uniques vus: {len(self.word_freq)}")

    def encode(self, text, max_len=50, add_special=True):
        tokens = self._tokenize(text)
        indices = [self.word2idx["<SOS>"]] if add_special else []

        for token in tokens:
            indices.append(self.word2idx.get(token, self.word2idx["<UNK>"]))

        if add_special:
            indices.append(self.word2idx["<EOS>"])

        if len(indices) > max_len:
            indices = indices[:max_len]
            if add_special:
                indices[-1] = self.word2idx["<EOS>"]
        else:
            indices.extend([self.word2idx["<PAD>"]] * (max_len - len(indices)))

        return indices

    def decode(self, indices, skip_special=True):
        words = []
        for idx in indices:
            word = self.idx2word.get(int(idx), "<UNK>")
            if skip_special and word in {"<PAD>", "<SOS>", "<EOS>"}:
                continue
            if word == "<EOS>":
                break
            words.append(word)
        return " ".join(words)

    def to_dict(self):
        return {
            "max_vocab_size": self.max_vocab_size,
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "word_freq": dict(self.word_freq),
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, data):
        tokenizer = cls(data.get("max_vocab_size", 10000))
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        tokenizer.word_freq = Counter(data.get("word_freq", {}))
        tokenizer.vocab_size = data["vocab_size"]
        return tokenizer


class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        effective_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, tgt_input):
        src_emb = self.embedding(src)
        _, hidden = self.encoder(src_emb)

        tgt_emb = self.embedding(tgt_input)
        decoder_output, _ = self.decoder(tgt_emb, hidden)
        return self.output(decoder_output)

    def encode(self, src):
        src_emb = self.embedding(src)
        _, hidden = self.encoder(src_emb)
        return hidden

    def decode_step(self, token, hidden):
        embedded = self.embedding(token)
        output, hidden = self.decoder(embedded, hidden)
        logits = self.output(output[:, -1, :])
        return logits, hidden

    def count_parameters(self):
        return sum(param.numel() for param in self.parameters())


class NeuralAI:
    """Assistant conversationnel entraine avec PyTorch."""

    def __init__(self, model_file="neural_brain.pkl"):
        self.model_file = model_file
        self.tokenizer = None
        self.model = None
        self.is_trained = False

        self.embedding_dim = 128
        self.hidden_size = 256
        self.num_layers = 2
        self.max_seq_len = 30
        self.learning_rate = 0.001
        self.batch_size = 32
        self.teacher_forcing_ratio = 1.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stats = {
            "epochs_trained": 0,
            "total_loss": [],
            "conversations": 0,
            "last_training": None,
            "device": str(self.device),
        }

        self.context = []
        self.load_model()

    def _build_model(self):
        self.model = Seq2SeqModel(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)

    def load_model(self):
        if not os.path.exists(self.model_file):
            print("[NEURAL] Nouveau cerveau - entrainement necessaire!")
            return

        try:
            checkpoint = torch.load(self.model_file, map_location=self.device)
            if not isinstance(checkpoint, dict) or "tokenizer" not in checkpoint:
                raise ValueError("format de checkpoint non reconnu")

            self.tokenizer = Tokenizer.from_dict(checkpoint["tokenizer"])
            self.stats.update(checkpoint.get("stats", {}))
            self.stats["device"] = str(self.device)
            self.is_trained = checkpoint.get("is_trained", False)

            self._build_model()
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()

            print("[NEURAL] Cerveau PyTorch charge!")
            print(f"   - Device: {self.device}")
            print(f"   - Epochs entraines: {self.stats['epochs_trained']}")
            if self.stats["total_loss"]:
                print(f"   - Derniere loss: {self.stats['total_loss'][-1]:.4f}")
        except Exception as error:
            print(f"[INFO] Ancien modele incompatible ou illisible: {error}")
            print("[INFO] Lance /train pour creer un nouveau modele PyTorch.")
            self.tokenizer = None
            self.model = None
            self.is_trained = False

    def save_model(self):
        if not self.model or not self.tokenizer:
            return

        checkpoint = {
            "tokenizer": self.tokenizer.to_dict(),
            "model_state": self.model.state_dict(),
            "stats": self.stats,
            "is_trained": self.is_trained,
            "config": {
                "embedding_dim": self.embedding_dim,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "max_seq_len": self.max_seq_len,
                "learning_rate": self.learning_rate,
            },
        }
        torch.save(checkpoint, self.model_file)

    def _prepare_tensors(self, qa_pairs):
        questions = [q for q, _ in qa_pairs]
        answers = [a for _, a in qa_pairs]
        all_texts = questions + answers

        print("\n[VOCABULAIRE] Construction...")
        self.tokenizer = Tokenizer(max_vocab_size=8000)
        self.tokenizer.fit(all_texts)

        print("\n[ENCODAGE] Preparation des donnees...")
        total = len(qa_pairs)
        src_rows = []
        tgt_rows = []

        for idx, (question, answer) in enumerate(qa_pairs):
            progress = (idx + 1) / total * 100
            bar_len = 30
            filled = int(bar_len * (idx + 1) / total)
            bar = "#" * filled + "-" * (bar_len - filled)
            sys.stdout.write(f"\r   [{bar}] {progress:.1f}% - Encodage...")
            sys.stdout.flush()

            src_rows.append(self.tokenizer.encode(question, self.max_seq_len))
            tgt_rows.append(self.tokenizer.encode(answer, self.max_seq_len))

        sys.stdout.write(f"\r   [{'#' * 30}] 100.0% - Termine!              \n")

        src = torch.tensor(src_rows, dtype=torch.long)
        tgt = torch.tensor(tgt_rows, dtype=torch.long)
        return src, tgt

    def train(self, qa_pairs, epochs=50, batch_size=None, verbose=True):
        print("\n" + "=" * 60)
        print("   ENTRAINEMENT DU RESEAU DE NEURONES")
        print("=" * 60)

        if batch_size is None:
            batch_size = self.batch_size

        src, tgt = self._prepare_tensors(qa_pairs)

        print("\n[MODELE] Initialisation du reseau PyTorch...")
        self._build_model()
        self.model.train()

        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.word2idx["<PAD>"])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        n_samples = src.size(0)
        n_batches = max(1, (n_samples + batch_size - 1) // batch_size)

        print(f"   - Device: {self.device}")
        print(f"   - Paires Q/R: {n_samples}")
        print(f"   - Batches par epoch: {n_batches}")
        print(f"   - Epochs: {epochs}")
        print(f"   - Learning rate: {self.learning_rate}")
        print(f"   - Parametres: {self.model.count_parameters():,}")

        print("\n[TRAINING] Debut de l'entrainement...\n")
        start_time = datetime.now()
        losses = []
        best_loss = float("inf")

        for epoch in range(epochs):
            epoch_start = datetime.now()
            permutation = torch.randperm(n_samples)
            src_shuffled = src[permutation]
            tgt_shuffled = tgt[permutation]
            epoch_loss = 0.0

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                src_batch = src_shuffled[start_idx:end_idx].to(self.device)
                tgt_batch = tgt_shuffled[start_idx:end_idx].to(self.device)

                tgt_input = tgt_batch[:, :-1]
                tgt_output = tgt_batch[:, 1:]

                optimizer.zero_grad()
                logits = self.model(src_batch, tgt_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss

                batch_progress = (batch_idx + 1) / n_batches * 100
                sys.stdout.write(
                    f"\r   Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{n_batches} ({batch_progress:.0f}%) | Loss: {batch_loss:.4f}"
                )
                sys.stdout.flush()

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            best_loss = min(best_loss, avg_loss)

            elapsed = (datetime.now() - start_time).total_seconds()
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1) if epoch + 1 else 0
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            progress_pct = (epoch + 1) / epochs * 100
            bar_len = 30
            filled = int(bar_len * (epoch + 1) / epochs)
            bar = "#" * filled + "-" * (bar_len - filled)

            sys.stdout.write("\r" + " " * 100 + "\r")
            print(
                f"   Epoch {epoch + 1:3d}/{epochs} [{bar}] {progress_pct:5.1f}% | "
                f"Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | {epoch_time:.1f}s/epoch | ETA: {eta:.0f}s"
            )

        print()
        total_time = (datetime.now() - start_time).total_seconds()
        self.stats["epochs_trained"] += epochs
        self.stats["total_loss"] = losses
        self.stats["last_training"] = datetime.now().isoformat()
        self.stats["device"] = str(self.device)

        print("=" * 60)
        print("   ENTRAINEMENT TERMINE!")
        print("=" * 60)
        print(f"   Duree totale:    {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"   Epochs:          {epochs}")
        print(f"   Paires Q/R:      {n_samples}")
        print(f"   Batches/epoch:   {n_batches}")
        print(f"   Loss initiale:   {losses[0]:.4f}")
        print(f"   Loss finale:     {losses[-1]:.4f}")
        print(f"   Meilleure loss:  {best_loss:.4f}")
        improvement = ((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] > 0 else 0.0
        print(f"   Amelioration:    {improvement:.1f}%")
        print(f"   Vitesse:         {total_time / epochs:.2f}s/epoch")
        print(f"   Device utilise:  {self.device}")
        print("=" * 60)

        self.is_trained = True
        self.model.eval()
        self.save_model()
        return losses

    def generate(self, input_text, max_len=30, temperature=0.8):
        if not self.is_trained or self.model is None or self.tokenizer is None:
            return "Je dois d'abord etre entraine! Utilise /train"

        self.model.eval()

        src = torch.tensor(
            [self.tokenizer.encode(input_text, self.max_seq_len)],
            dtype=torch.long,
            device=self.device,
        )

        hidden = self.model.encode(src)
        current_token = torch.tensor([[self.tokenizer.word2idx["<SOS>"]]], dtype=torch.long, device=self.device)
        generated = []

        with torch.no_grad():
            for _ in range(max_len):
                logits, hidden = self.model.decode_step(current_token, hidden)
                logits = logits / max(temperature, 0.1)

                blocked = [
                    self.tokenizer.word2idx["<PAD>"],
                    self.tokenizer.word2idx["<UNK>"],
                    self.tokenizer.word2idx["<SOS>"],
                ]
                logits[:, blocked] = -1e9
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = int(next_token.item())

                if token_id == self.tokenizer.word2idx["<EOS>"]:
                    break

                generated.append(token_id)
                current_token = next_token

        return self.tokenizer.decode(generated)

    def respond(self, user_input):
        self.stats["conversations"] += 1
        self.context.append(("user", user_input))

        response = self.generate(user_input)
        if response:
            response = response.strip()
            if response and response[0].islower():
                response = response[0].upper() + response[1:]
            if response and response[-1] not in ".!?":
                response += "."
        else:
            response = random.choice([
                "Je reflechis encore.",
                "Peux-tu reformuler?",
                "Dis-m'en un peu plus.",
            ])

        self.context.append(("ai", response))
        if len(self.context) > 10:
            self.context = self.context[-10:]

        return response


def load_training_data_from_file(filepath="training_data.txt"):
    """Charge les donnees d'entrainement depuis un fichier externe."""
    qa_pairs = []

    if not os.path.exists(filepath):
        print(f"[ERREUR] Fichier non trouve: {filepath}")
        return []

    print(f"\n[CHARGEMENT] {filepath}")
    start_time = datetime.now()

    with open(filepath, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    total_lines = len(lines)
    count = 0

    for idx, line in enumerate(lines):
        progress = (idx + 1) / total_lines * 100 if total_lines else 100
        bar_len = 30
        filled = int(bar_len * (idx + 1) / total_lines) if total_lines else bar_len
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r   [{bar}] {progress:.1f}% - Lecture...")
        sys.stdout.flush()

        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "|" in line:
            question, answer = line.split("|", 1)
            question = question.strip()
            answer = answer.strip()
            if question and answer:
                qa_pairs.append((question, answer))
                count += 1

    load_time = (datetime.now() - start_time).total_seconds()
    sys.stdout.write(f"\r   [{'#' * 30}] 100.0% - Termine!              \n")
    print(f"   -> {count} paires Q/R chargees en {load_time:.2f}s")
    return qa_pairs


def main():
    print()
    print("=" * 70)
    print("   NEURAL AI - Intelligence Artificielle par Reseaux de Neurones v4.0")
    print("=" * 70)
    print()
    print("   Architecture: Seq2Seq LSTM avec PyTorch")
    print("   Acceleration: GPU automatique si CUDA est disponible")
    print()
    print("   Commandes:")
    print("   /train [epochs] [fichier]  - Entraine le reseau")
    print("                               (defaut: 50 epochs, training_base.txt)")
    print("   /stats                     - Affiche les statistiques")
    print("   /reset                     - Reinitialise le cerveau")
    print("   /quit                      - Quitter")
    print()
    print("   Ou pose simplement une question!")
    print("-" * 70)

    ai = NeuralAI()

    if not ai.is_trained:
        print()
        print("[INFO] Le reseau n'est pas entraine.")
        print("       Tape /train pour commencer l'entrainement.")
        print("       (Le GPU sera utilise automatiquement s'il est disponible)")
        print()

    while True:
        try:
            user_input = input("\nToi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[BYE] Au revoir!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("[BYE] Au revoir!")
            ai.save_model()
            break

        if user_input.lower().startswith("/train"):
            parts = user_input.split()
            epochs = 50
            if len(parts) > 1:
                try:
                    epochs = int(parts[1])
                except ValueError:
                    pass

            data_file = "training_base.txt"
            if len(parts) > 2:
                data_file = parts[2]

            qa_pairs = load_training_data_from_file(data_file)
            if not qa_pairs:
                print("[ERREUR] Aucune donnee d'entrainement trouvee!")
                continue

            print(f"\n[INFO] Entrainement avec {len(qa_pairs)} paires Q/R")
            print(f"       Epochs: {epochs}")
            print(f"       Device cible: {ai.device}")
            print("       Cela peut prendre plusieurs minutes...\n")
            ai.train(qa_pairs, epochs=epochs)
            continue

        if user_input.lower() == "/stats":
            print("\n[STATS] Statistiques du reseau:")
            print(f"   - Entraine: {'Oui' if ai.is_trained else 'Non'}")
            print(f"   - Device: {ai.device}")
            print(f"   - Epochs totaux: {ai.stats['epochs_trained']}")
            print(f"   - Conversations: {ai.stats['conversations']}")
            if ai.stats["total_loss"]:
                print(f"   - Derniere loss: {ai.stats['total_loss'][-1]:.4f}")
            if ai.stats["last_training"]:
                print(f"   - Dernier entrainement: {ai.stats['last_training']}")
            if ai.tokenizer:
                print(f"   - Vocabulaire: {ai.tokenizer.vocab_size} mots")
            if ai.model:
                print(f"   - Parametres: {ai.model.count_parameters():,}")
            continue

        if user_input.lower() == "/reset":
            if os.path.exists(ai.model_file):
                os.remove(ai.model_file)
            ai = NeuralAI()
            print("[OK] Cerveau reinitialise!")
            continue

        response = ai.respond(user_input)
        print(f"IA: {response}")


if __name__ == "__main__":
    main()
