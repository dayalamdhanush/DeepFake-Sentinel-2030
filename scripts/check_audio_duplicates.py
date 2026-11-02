from pathlib import Path
import hashlib
from collections import defaultdict

root = Path("datasets/audio")
hash_groups = defaultdict(list)

for split in ["train", "val", "test"]:
    for label in ["fake", "real"]:
        folder = root / split / label
        for file in folder.rglob("*.wav"):
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            hash_groups[digest].append(f"{split}/{label}/{file.name}")

duplicates = {h: files for h, files in hash_groups.items() if len(files) > 1}

print("Total unique audio files:", len(hash_groups))
print("Duplicate groups:", len(duplicates))

for files in duplicates.values():
    print("\nDUPLICATE:")
    for file in files:
        print(" ", file)
