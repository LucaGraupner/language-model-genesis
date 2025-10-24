# prepare.py
# Dieser Code lädt den 'wikitext'-Datensatz von Hugging Face herunter,
# bereitet ihn für das Training vor und speichert ihn in zwei binären Dateien: train.bin und val.bin.

import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import tiktoken # Wir verwenden den GPT-2 Tokenizer für eine bessere Qualität

# --- Hauptskript ---
if __name__ == '__main__':
    # Lädt den Datensatz 'wikitext' in der Konfiguration 'wikitext-103-raw-v1' von Hugging Face.
    # Dieser Datensatz ist eine saubere Sammlung von Wikipedia-Artikeln.
    print("Lade den Datensatz-Index für 'wikitext'...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split='train')

    # Filtere leere oder sehr kurze Zeilen heraus, die im Datensatz vorkommen können.
    dataset = dataset.filter(lambda example: len(example['text']) > 10)

    # Teilt den Datensatz in 99.5% Trainingsdaten und 0.5% Validierungsdaten auf.
    split_dataset = dataset.train_test_split(test_size=0.005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # Umbenennung von 'test' zu 'val'

    # Initialisiere den GPT-2 Tokenizer von tiktoken
    enc = tiktoken.get_encoding("gpt2")

    def process(example):
        # Nutze den Tokenizer, um den Text in eine Sequenz von Zahlen (Token-IDs) umzuwandeln.
        ids = enc.encode_ordinary(example['text'])
        # Füge das "End of Text"-Token hinzu, damit das Modell lernt, wo ein Dokument endet.
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Führe die Tokenisierung auf dem gesamten Datensatz aus.
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing the splits",
        num_proc=os.cpu_count(), # Nutze alle verfügbaren CPU-Kerne
    )

    # Schreibe die tokenisierten Daten in die .bin-Dateien.
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # GPT-2 Tokenizer hat ca. 50k Tokens, uint16 reicht dafür aus
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        idx = 0
        for example in tqdm(dset, desc=f"Schreibe {filename}"):
            arr[idx : idx + example['len']] = example['ids']
            idx += example['len']
        arr.flush()

    print("Datenaufbereitung abgeschlossen.")