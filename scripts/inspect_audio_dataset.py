from pathlib import Path
from collections import Counter
import librosa
import numpy as np

root = Path("datasets/audio")

for split in ["train", "val", "test"]:
    print(f"\n========== {split.upper()} ==========")

    for label in ["fake", "real"]:
        folder = root / split / label
        files = list(folder.glob("*.wav"))

        print(f"\n{label.upper()}: {len(files)} files")

        for path in files[:2]:
            y, sr = librosa.load(path, sr=16000, mono=True)
            mel = librosa.feature.melspectrogram(
                y=y[:48000] if len(y) >= 48000 else np.pad(y, (0, 48000-len(y))),
                sr=16000,
                n_mels=128,
                n_fft=1024,
                hop_length=512
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)

            print(
                path.name,
                "| Duration:", round(len(y)/sr, 2),
                "| Mean:", round(float(log_mel.mean()), 3),
                "| Std:", round(float(log_mel.std()), 3),
                "| Min:", round(float(log_mel.min()), 3),
                "| Max:", round(float(log_mel.max()), 3)
            )
