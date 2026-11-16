# Smart Text-to-Speech App, Video Narration Pipeline & MCP Server

A multi-engine Python Text-to-Speech suite featuring:

- **MCP Server (`tts-mcp`)** — Native Model Context Protocol (stdio JSON-RPC 2.0) server so AI agents can discover voices, synthesize speech, generate frame-accurate video narration & captions, mux audio/subtitles onto MP4s, and enroll or fine-tune custom user voices.
- **Docker & Docker Compose Ready** — Combined `Dockerfile` + `docker-compose.yml` bundling Python, `ffmpeg`, `espeak-ng`, `edge-tts`, `pyttsx3`, and optional CPU/CUDA Coqui XTTS v2 support.
- **Video Generation Pipeline** — Converts scripts or multi-scene storyboards into per-scene audio clips, a loudness-normalized master track (`-16 LUFS`), `.srt` & `.vtt` subtitles, and frame-accurate `.timings.json` manifests ready for **Remotion**, **HyperFrames**, and **FFmpeg**.
- **Voice Personalization & Training** — Users can **enroll fast reusable voice profiles** from short samples OR **fine-tune (train) XTTS v2 on their own voice** dataset.
- **Desktop GUI (PyQt5)** & **CLI (`tts-cli`)** — Interactive playback, microphone recording, speed control, voice profile management, and audio/subtitle export.

---

## Engines

| Engine | Mode | Strengths |
| --- | --- | --- |
| **`edge-tts`** | Online | 300+ natural neural voices across 75+ languages + native `WordBoundary` timestamps for accurate captions |
| **`xtts_v2`** | Offline (Local AI) | Zero-shot voice cloning, saved **enrolled voice profiles**, **custom fine-tuned user voice models**, and 58 built-in neural speakers across 17 languages |
| **`pyttsx3`** | Offline (OS) | Ultra-lightweight fallback using system/eSpeak voices with zero model downloads |
| **`auto` (Smart Router)** | Hybrid | Picks the best engine automatically (XTTS when a user voice profile/sample is provided, otherwise `edge-tts` with automatic offline fallback) |

---

## Minimum Hardware Requirements

| Tier / Feature | CPU | RAM | Disk Space | GPU / VRAM | Extra Tools |
| --- | --- | --- | --- | --- | --- |
| **1. Standard TTS & Video Narration** (`edge-tts`, `pyttsx3`, MCP server) | Dual-core 64-bit | **512 MB** (1 GB rec.) | **200 MB** | None required | `ffmpeg` + `ffprobe` recommended for WAV/MP3 transcoding, `-16 LUFS` loudness, and video muxing |
| **2. Local Voice Cloning & Enrolled Profiles** (`XTTS v2` inference) | Quad-core w/ AVX2 | **8 GB min** (16 GB rec.) | **5 GB** (~2 GB PyTorch + 1.8 GB XTTS weights) | Optional (CPU works slower; **NVIDIA CUDA 4 GB+ VRAM** or Apple MPS recommended) | `requirements-xtts.txt` |
| **3. Custom Voice Training / Fine-Tuning** (`train_custom_voice`) | Quad-core+ | **16 GB min** (32 GB rec.) | **10–15 GB free** | **NVIDIA CUDA GPU with >= 8 GB VRAM required** (12 GB+ rec.) | CUDA PyTorch + `requirements-xtts.txt` (+ optional `faster-whisper` for auto-transcription) |

> Check your machine's live readiness anytime with `tts-cli status` or the MCP tool `tts_status`.

---

## Setup & Installation

### Option 1: Docker & Docker Compose (Recommended for MCP & Headless Usage)

The [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml) combine all system (`ffmpeg`, `espeak-ng`, `libsndfile1`) and Python dependencies in one reproducible container with persistent volumes for voice profiles (`/data`) and cached XTTS weights (`/root/.local/share/tts`):

```bash
# 1. Default lightweight MCP image (edge-tts + pyttsx3 + ffmpeg):
docker compose build tts-mcp
docker compose run --rm tts-mcp tts-cli status

# 2. Optional CPU XTTS v2 image (local voice cloning & profile enrollment):
docker compose --profile xtts build tts-mcp-xtts
docker compose --profile xtts run --rm tts-mcp-xtts tts-cli status

# 3. Optional NVIDIA CUDA GPU image (fast cloning & custom voice training):
# Requires NVIDIA Container Toolkit on the host.
docker compose --profile gpu build tts-mcp-gpu
docker compose --profile gpu run --rm tts-mcp-gpu tts-cli status
```

Set `TTS_WORKSPACE=/path/to/your/project` (defaults to `.`) so generated audio, subtitles, `.timings.json`, and muxed videos map directly into `/workspace` on the host.

### Option 2: Local Python Virtual Environment

#### Core Installation (GUI, CLI, MCP Server, `edge-tts`, `pyttsx3`)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps
```

Run the desktop app:

```bash
python main.py
# or: tts-app
```

#### Optional: Local Voice Cloning & Voice Training (Coqui XTTS v2)

```bash
# CPU-only (good for voice profile enrollment and occasional local synthesis):
pip install torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-xtts.txt

# OR NVIDIA CUDA GPU (recommended for fast cloning, required for custom voice training):
# Install CUDA torch/torchaudio/torchcodec from https://pytorch.org/get-started/locally/
pip install -r requirements-xtts.txt

# Optional: auto-transcribe voice training clips when you don't have transcripts:
pip install faster-whisper
```

---

## How to Use the MCP Server

This repository exposes a zero-dependency **Model Context Protocol (MCP)** server over stdio JSON-RPC 2.0 (`tts-mcp` / `python -m tts_app.mcp_server`), making it plug-and-play for **Antigravity**, **Claude Desktop**, **Cursor**, and video-generation agents (**Remotion**, **HyperFrames**, **FFmpeg**).

### 1. Register the MCP Server in Your AI Client

**Using Docker Compose (zero Python setup on host):**

> Note the `-i -T` flags: `-i` keeps stdin open for JSON-RPC requests and `-T` disables pseudo-TTY allocation so stdout carries clean JSON-RPC lines.

```json
{
  "mcpServers": {
    "tts": {
      "command": "docker",
      "args": [
        "compose",
        "-f",
        "/path/to/tts/docker-compose.yml",
        "run",
        "--rm",
        "-i",
        "-T",
        "tts-mcp"
      ],
      "env": {
        "TTS_WORKSPACE": "/path/to/your/video-project"
      }
    }
  }
}
```

*(Swap `"tts-mcp"` for `"tts-mcp-xtts"` or `"tts-mcp-gpu"` if using the `--profile xtts` / `--profile gpu` images.)*

**Using a local Python `venv`** (also in [`configs/mcp-client-example.json`](configs/mcp-client-example.json)):

```json
{
  "mcpServers": {
    "tts": {
      "command": "/path/to/tts/venv/bin/python",
      "args": ["-m", "tts_app.mcp_server"]
    }
  }
}
```

### 2. Typical AI Agent Workflow

1. **Check capabilities (`tts_status`):** Inspect available engines (`edge-tts`, `xtts_v2`, `pyttsx3`), GPU/VRAM readiness, `ffmpeg`, and saved custom voice profiles.
2. **Pick a voice (`list_voices`):** Filter by `engine`, `language` (e.g. `"en"`, `"ar"`, `"fr"`), `gender`, or `search`, or choose an enrolled user voice profile.
3. **Generate single audio + captions (`synthesize_speech`):** Pass `text`, optional `voice_id` / `voice_profile` / `reference_audio`, `format` (`"mp3"` or `"wav"`), `normalize_loudness: true`, and `generate_subtitles: true`.
4. **Generate multi-scene video narration (`generate_video_voiceover`):** Pass a `script` or structured `scenes` array + target `fps` (`24`/`30`/`60`) to output:
   - Loudness-normalized (`-16 LUFS`) master audio track
   - Individual per-scene clips (`scenes/*.wav`)
   - `.srt`, `.vtt`, and frame-accurate `.timings.json` (`start_frame`, `end_frame`, `words`, `cues`)
   - Optional muxed MP4 if `video_path` is provided (or call `mux_audio_to_video` afterward).
5. **Personalize with the user's voice (`enroll_voice_profile` / `train_custom_voice`):** Enroll a fast reusable profile or fine-tune XTTS v2, then pass `voice_profile` in any narration call.

### 3. Quick Terminal Smoke Test of the MCP Protocol

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"tts_status","arguments":{}}}' \
  | ./venv/bin/python -m tts_app.mcp_server
```

### Exposed MCP Tools, Resources & Prompts

| Tool | Purpose |
| --- | --- |
| `tts_status` | Inspect available engines, GPU/VRAM/RAM/disk hardware readiness, `ffmpeg`, and saved user profiles |
| `list_voices` | Query voices by `engine`, `language`, `gender`, or `search`, plus saved custom voice profiles |
| `synthesize_speech` | Generate MP3/WAV speech (using built-in voices or a user voice profile) + optional `.srt`, `.vtt`, and `.timings.json` |
| `generate_video_voiceover` | **Video agent tool**: turns a multi-scene storyboard or script into per-scene audio clips, a stitched `-16 LUFS` master track, `.srt`/`.vtt` captions, frame-accurate `.timings.json` (`fps`), and optional MP4 muxing |
| `mux_audio_to_video` | Attach generated narration (and soft or burned-in `.srt` subtitles) onto an MP4/MOV/WEBM video via FFmpeg |
| `enroll_voice_profile` | Preprocess 6–30s+ user recordings and save reusable XTTS conditioning latents as a named profile |
| `train_custom_voice` | Prepare a dataset and fine-tune XTTS v2 GPT weights on a user's voice (supports `dry_run=true` pre-flight validation) |
| `manage_voice_profiles` | List, inspect, or delete saved enrolled and fine-tuned voice profiles |

- **Resources:** `tts://status`, `tts://voices/profiles`, `tts://guides/hardware-and-training`
- **Prompts:** `video_narration_script`, `voice_training_checklist`

---

## Video Generation Workflow (CLI)

Produce narration tracks and synced frame timings in one command:

```bash
tts-cli video-voiceover \
  --script "[scene: hook]\nStop editing captions manually.\n\n[scene: demo]\nGenerate voiceover and frame timelines together." \
  --output-dir ./outputs/launch_video \
  --name narration \
  --fps 30 \
  --format mp3
```

Outputs created in `./outputs/launch_video/`:

- `narration.mp3` — Loudness-normalized (`-16 LUFS`), silence-trimmed master voiceover
- `scenes/001_hook.wav`, `scenes/002_demo.wav` — Individual per-scene clips
- `narration.srt` & `narration.vtt` — Video caption tracks
- `narration.timings.json` — Scene start/end in **milliseconds and exact frames**, plus cue & word timings (`start_frame`, `end_frame`) ready for Remotion (`@remotion/captions`) or HyperFrames timelines

---

## Training & Personalizing Your Own Voice

You can use **your own voice** in two ways depending on your hardware and dataset size:

### Option A: Fast Voice Profile Enrollment (6–30s, CPU or GPU)

Preprocesses one or more clean voice recordings (22.05 kHz mono, 70 Hz high-pass, silence-trimmed, normalized) and caches XTTS speaker conditioning latents (`gpt_cond_latent` and `speaker_embedding`) for fast reuse:

- **Desktop GUI:** Select **XTTS v2 (your voice & local)** -> click **Record** or **Choose voice sample…** -> click **Save as Profile…**
- **CLI:**
  ```bash
  tts-cli enroll-voice --name "my_voice" --audio sample1.wav sample2.wav --language en
  tts-cli speak "Hello from my voice profile!" -o hello.mp3 --voice-profile my_voice --subtitles
  ```
- **MCP:** Call `enroll_voice_profile` then pass `voice_profile: "my_voice"` to `synthesize_speech` or `generate_video_voiceover`.

### Option B: Full Custom Voice Fine-Tuning (Minutes of Audio, CUDA GPU >= 8 GB VRAM)

Fine-tunes the XTTS v2 GPT weights on your own speech dataset for stronger accent, cadence, and timbre consistency:

1. Prepare **>= 2 minutes** (10–30 minutes recommended) of clean, mono speech clips (2–11 seconds each) with either:
   - a pipe/comma-separated `metadata.csv` (`audio_file|text`), OR
   - a folder of audio clips + `faster-whisper` installed (`--auto-transcribe`).
2. Run a pre-flight dry-run check, then train:
   ```bash
   tts-cli train-voice --name "my_trained_voice" --metadata-csv ./my_dataset/metadata.csv --language en --dry-run
   tts-cli train-voice --name "my_trained_voice" --metadata-csv ./my_dataset/metadata.csv --language en --epochs 10
   ```
3. Once finished, your fine-tuned model is automatically registered as a voice profile (`--voice-profile my_trained_voice` or `voice_profile: "my_trained_voice"` in MCP).

> XTTS v2 model weights use the [Coqui Public Model License](https://coqui.ai/cpml) (non-commercial).

---

## Project Structure

```text
.
├─ Dockerfile                    # Unified container image (ffmpeg, espeak-ng, core + optional XTTS/CUDA)
├─ docker-compose.yml            # Services for tts-mcp (default), tts-mcp-xtts, and tts-mcp-gpu
├─ .dockerignore
├─ main.py                       # Desktop GUI launcher
├─ pyproject.toml                # Package metadata, entry points (tts-app, tts-cli, tts-mcp)
├─ requirements.txt              # Core dependencies
├─ requirements-xtts.txt         # Optional local XTTS v2 cloning & training deps
├─ configs/
│  └─ mcp-client-example.json    # Ready-to-copy local Python MCP client configuration
├─ tts_app/
│  ├─ models.py                  # Voice, WordTiming, CaptionCue, SynthesisResult, VoiceProfile, TtsError
│  ├─ audio_utils.py             # FFmpeg/FFprobe + stdlib WAV probing, transcode, loudnorm, trim, concat, video mux
│  ├─ subtitles.py               # SRT, VTT, and frame-accurate JSON timing manifests
│  ├─ voice_profiles.py          # Hardware checks, voice profile enrollment, XTTS dataset prep & fine-tuning
│  ├─ tts_engine.py              # EdgeTtsEngine, XttsEngine, Pyttsx3Engine, and smart auto-router
│  ├─ video_pipeline.py          # Multi-scene storyboard narration & timeline builder
│  ├─ mcp_server.py              # Stdio JSON-RPC 2.0 MCP server
│  ├─ cli.py                     # CLI interface (`tts-cli`)
│  ├─ audio_player.py            # PyQt5 QMediaPlayer wrapper
│  └─ gui.py                     # PyQt5 desktop application
├─ tests/                        # Automated pytest suite
├─ tts_app.spec                  # PyInstaller build configuration
└─ build_windows.bat             # One-click Windows .exe build script
```

---

## Development & Tests

```bash
pip install -e ".[dev]"
pytest -v
ruff check tts_app tests main.py
```

### Building a Standalone Windows `.exe`

Run `build_windows.bat` on a Windows machine with Python 3.10+ to produce `dist\TTSApp.exe` (bundles `edge-tts` and `pyttsx3`; keeps heavy XTTS/PyTorch optional).
