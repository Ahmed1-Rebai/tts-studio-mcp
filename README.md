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
├─ tts_app.spec             # PyInstaller config (builds the single-file .exe)
├─ build_windows.bat        # one-click Windows installer build script
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

## Building a standalone .exe (for a friend's Windows laptop)

An executable only runs on the OS it was built *on*. I've provided everything
needed to produce a single-file, no-install Windows `.exe`. Take this whole
folder to a Windows machine and:

1. Copy the project folder (it contains `tts_app.spec`, `build_windows.bat`,
   `requirements.txt`, and the `tts_app/` package).
2. Double-click **`build_windows.bat`**. It will:
   - create a venv,
   - install the dependencies plus PyInstaller,
   - build `dist\TTSApp.exe` using `tts_app.spec`.
3. Grab **`dist\TTSApp.exe`** — copy it onto your friend's computer and run it.
   No Python or installation is required on their machine.

> The `.spec` already bundles the PyQt5 plugins and all edge-tts/pyttsx3
> dependencies, and builds with the console window hidden.