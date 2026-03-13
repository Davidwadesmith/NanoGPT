import json
import numpy as np
import os
from tqdm import tqdm

EOS_TOKEN = "<eos>"
DEFAULT_SAMPLE_SEPARATOR = "<|endoftext|>"


def stream_data_prepare(
    train_file: str,
    validate_file: str,
    chunk_size: int = 4096,
    sample_separator: str = DEFAULT_SAMPLE_SEPARATOR,
):
    char_set = set()
    with open(train_file, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            for c in chunk:
                char_set.add(c)
    with open(validate_file, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            for c in chunk:
                char_set.add(c)

    char_set.add(EOS_TOKEN)
    
    char_list = sorted(list(char_set))
    char2id = {c : i for i, c in enumerate(char_list)}
    id2char = {i: c for c, i in char2id.items()}
    
    vocab_size = len(char_list)

    meta = {"vocab size": vocab_size, "char to id": char2id, "id to char": id2char}
    meta["eos token"] = EOS_TOKEN
    meta["eos id"] = char2id[EOS_TOKEN]
    
    with open("./data/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    

    train_bin = open("./data/train.bin", "wb")
    validate_bin = open("./data/validate.bin", "wb")
    write_encoded_file(train_file, train_bin, char2id, chunk_size, sample_separator)
    write_encoded_file(
        validate_file, validate_bin, char2id, chunk_size, sample_separator
    )
    train_bin.close()
    validate_bin.close()


def write_encoded_file(
    input_file: str,
    output_file,
    char2id: dict[str, int],
    chunk_size: int,
    sample_separator: str,
):
    eos_id = char2id[EOS_TOKEN]
    separator_len = len(sample_separator)
    carry = ""
    wrote_anything = False

    with tqdm(total=os.path.getsize(input_file), desc=f"Encoding {input_file}") as pbar:
        with open(input_file, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                carry += chunk
                safe_len = max(0, len(carry) - max(separator_len - 1, 0))
                ids, carry = encode_text_chunk(carry, char2id, eos_id, sample_separator, safe_len)
                if ids:
                    output_file.write(np.array(ids, dtype=np.int32).tobytes())
                    # Update pbar by the number of bytes read from the input file, not the number of tokens written.
                    pbar.update(len(bytes(carry[:safe_len], encoding="utf-8")))
                    wrote_anything = True

        ids, _ = encode_text_chunk(carry, char2id, eos_id, sample_separator, len(carry))
        if ids:
            output_file.write(np.array(ids, dtype=np.int32).tobytes())
            pbar.update(len(carry))
            wrote_anything = True

        if wrote_anything:
            output_file.write(np.array([eos_id], dtype=np.int32).tobytes())
            pbar.update(1)


def encode_text_chunk(
    text: str,
    char2id: dict[str, int],
    eos_id: int,
    sample_separator: str,
    limit: int,
):
    ids = []
    idx = 0
    separator_len = len(sample_separator)

    while idx < limit:
        if separator_len > 0 and text.startswith(sample_separator, idx):
            ids.extend(char2id[char] for char in sample_separator)
            ids.append(eos_id)
            idx += separator_len
            continue

        ids.append(char2id[text[idx]])
        idx += 1

    return ids, text[idx:]

if __name__ == "__main__":
    stream_data_prepare("./data/TinyStories-train.txt", "./data/TinyStories-valid.txt")