# MidiForge — Model Training

Repository with notebooks and scripts used to preprocess data and train the LSTM models used by the backend.

## Purpose
Prepare token dictionaries, generate seed files and train per-genre LSTM models saved as .pt for inference.

## Contents
- src_new/
  - model.py — model definition and training utilities
  - midi.py — MIDI handling / preprocessing
  - generate_seeds.ipynb, final_preprocess.ipynb, train_all_separately.ipynb — notebooks used during development
- Outputs expected:
  - models/*.pt
  - models/word_to_idx.json
  - models/idx_to_word.json
  - seeds/*_seeds.txt

## Quick start
1. Create env and install deps (use included requirements for training if provided):
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run notebooks or training scripts in src_new to reproduce models.

## How to use outputs
- Copy `models/` and `seeds/` into the backend repository root so backend can load models and seeds for generation.

## Notes
- Training requires GPU for reasonable speed; adjust model/training settings in model.py.
- Keep vocab files consistent: backend expects `word_to_idx.json` and `idx_to_word.json`.

## License
See [LICENSE](LICENSE) file.
