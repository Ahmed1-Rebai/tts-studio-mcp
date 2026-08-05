# Text-to-Speech Desktop App

A Python desktop application (PyQt5) that turns text files — or text pasted into the
window — into spoken audio with Play / Pause / Stop / Export controls.

## Features

- Load a `.txt` file or paste/type text directly.
- Choose a TTS **engine** (catalogued dynamically):
  - **edge-tts (online)** — default. Huge multilingual catalog (75+ languages),
    clear male/female voices per language, natural neural quality.
  - **pyttsx3 (offline)** — fully offline, uses the OS/espeak system voices.
    Used as a fallback when there is no internet.
- Choose a **language** (dropdown populated from the selected engine).
- Filter voices by **gender** (Male / Female) and **age** (Adult / Child) when
  the engine exposes that metadata.
- **Play / Pause / Stop** with a progress slider and elapsed/total time.
- **Export** the generated speech to **MP3** or **WAV**.
- Clear, friendly error dialogs for unsupported language, empty text, or engine
  failure; a network-free engine, etc.

## Project structure

```
.
├─ main.py                  # entry point (creates window + event loop)
├─ requirements.txt
└─ tts_app/
   ├─ tts_engine.py           # Voice/TtsError + engine abstraction, edge-tts & pyttsx3
   ├─ audio_player.py          # QtMultimedia QMediaPlayer wrapper (progress)
   │                           #    + signal-based state relayed to GUI
   └─ gui.py                   # MainWindow: all widgets & drop-down logic
```

## Setup & run

```bash
python -m venv venv
source venv/bin/activate          # (or venv\Scripts\activate on Windows)
pip install -r requirements.txt
python main.py
```

## Command-line smoke test

The engine layer can be exercised without the GUI:

```bash
python -m tts_app.tts_engine "Hello from the CLI"
```

This enumerates the voice catalog, prints a sample of voices, and writes a test
MP3 to the system temp directory.