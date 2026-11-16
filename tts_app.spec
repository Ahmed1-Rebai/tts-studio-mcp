# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Text-to-Speech desktop app.

Build on the *target* operating system (this spec is cross-platform). On Windows
run:  pyinstaller --noconfirm tts_app.spec
to produce a single-file, windowed executable in ./dist.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = [
    "tts_app.models",
    "tts_app.audio_utils",
    "tts_app.subtitles",
    "tts_app.voice_profiles",
    "tts_app.audio_player",
    "tts_app.tts_engine",
    "tts_app.video_pipeline",
    "tts_app.mcp_server",
    "tts_app.cli",
    "tts_app.gui",
]
# edge-tts uses aiohttp/async with dynamically loaded submodules.
hiddenimports += collect_submodules("edge_tts")
hiddenimports += collect_submodules("aiohttp")
hiddenimports += collect_submodules("aiohappyeyeballs")
hiddenimports += collect_submodules("multidict")
hiddenimports += collect_submodules("yarl")
hiddenimports += collect_submodules("pyttsx3")
hiddenimports += collect_submodules("pydantic")
hiddenimports += collect_submodules("anyio")

datas = []
datas += collect_data_files("certifi")
datas += collect_data_files("edge_tts")
datas += collect_data_files("pyttsx3")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Keep the optional XTTS voice-cloning/training stack out of the binary — it
    # is imported lazily at runtime and would add ~2 GB if bundled.
    excludes=[
        "tkinter",
        "torch",
        "torchaudio",
        "torchcodec",
        "torchvision",
        "TTS",
        "trainer",
        "transformers",
        "faster_whisper",
        "librosa",
        "numba",
        "soundfile",
        "scipy",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="TTSApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
