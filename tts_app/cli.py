"""Command-line interface for synthesis, video voiceovers, voice training, and MCP."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import typing as _t

from tts_app.models import TtsError
from tts_app.tts_engine import available_engines, get_engine, smart_synthesize
from tts_app.video_pipeline import generate_video_voiceover
from tts_app.voice_profiles import (
    check_hardware_requirements,
    delete_profile,
    enroll_voice_profile,
    get_profile,
    list_profiles,
    train_custom_voice,
)


def _print_json(data: _t.Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts-cli",
        description="Smart TTS CLI: speech synthesis, video voiceovers, voice training, and MCP server.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = sub.add_parser("status", help="Show engines, hardware tiers, and saved voice profiles.")
    p_status.add_argument("--json", action="store_true", help="Output JSON.")

    # voices
    p_voices = sub.add_parser("voices", help="List available voices.")
    p_voices.add_argument("--engine", default="all")
    p_voices.add_argument("--language", default="")
    p_voices.add_argument("--gender", default="")
    p_voices.add_argument("--search", default="")
    p_voices.add_argument("--limit", type=int, default=40)
    p_voices.add_argument("--json", action="store_true")

    # speak
    p_speak = sub.add_parser("speak", help="Synthesize speech to MP3/WAV (plus optional subtitles).")
    p_speak.add_argument("text", help="Text to speak (or path if --text-file is used).")
    p_speak.add_argument("--text-file", action="store_true", help="Treat `text` argument as a file path.")
    p_speak.add_argument("-o", "--output", required=True, help="Output .mp3 or .wav path.")
    p_speak.add_argument("--engine", default="auto")
    p_speak.add_argument("--language", default="en")
    p_speak.add_argument("--voice", default="")
    p_speak.add_argument("--gender", default="")
    p_speak.add_argument("--voice-profile", default=None)
    p_speak.add_argument("--reference-audio", default=None)
    p_speak.add_argument("--speed", type=float, default=1.0)
    p_speak.add_argument("--pitch", default="+0Hz")
    p_speak.add_argument("--subtitles", action="store_true", help="Write .srt, .vtt, and .timings.json.")
    p_speak.add_argument("--fps", type=int, default=30)
    p_speak.add_argument("--normalize", action="store_true")
    p_speak.add_argument("--trim", action="store_true")
    p_speak.add_argument("--offline", action="store_true")
    p_speak.add_argument("--json", action="store_true")

    # video-voiceover
    p_vid = sub.add_parser("video-voiceover", help="Generate multi-scene narration, SRT/VTT, and timeline JSON.")
    src_grp = p_vid.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--script", help="Inline script text (split on blank lines or '[scene: id]').")
    src_grp.add_argument("--script-file", help="Path to a .txt narration script.")
    p_vid.add_argument("--output-dir", required=True)
    p_vid.add_argument("--name", default="voiceover")
    p_vid.add_argument("--format", default="mp3", choices=["mp3", "wav"])
    p_vid.add_argument("--fps", type=int, default=30)
    p_vid.add_argument("--engine", default="auto")
    p_vid.add_argument("--language", default="en")
    p_vid.add_argument("--voice", default="")
    p_vid.add_argument("--gender", default="")
    p_vid.add_argument("--voice-profile", default=None)
    p_vid.add_argument("--reference-audio", default=None)
    p_vid.add_argument("--speed", type=float, default=1.0)
    p_vid.add_argument("--pause-ms", type=int, default=350)
    p_vid.add_argument("--no-trim", action="store_true")
    p_vid.add_argument("--no-normalize", action="store_true")
    p_vid.add_argument("--video", default=None, help="Optional video file to mux narration onto.")
    p_vid.add_argument("--burn-subtitles", action="store_true")
    p_vid.add_argument("--json", action="store_true")

    # enroll-voice
    p_enroll = sub.add_parser("enroll-voice", help="Enroll a fast reusable XTTS voice profile from audio clips.")
    p_enroll.add_argument("--name", required=True)
    p_enroll.add_argument("--audio", nargs="+", required=True, help="Audio file(s) or directory.")
    p_enroll.add_argument("--language", default="en")
    p_enroll.add_argument("--no-preprocess", action="store_true")
    p_enroll.add_argument("--notes", default="")

    # train-voice
    p_train = sub.add_parser("train-voice", help="Fine-tune XTTS v2 on a custom voice dataset.")
    p_train.add_argument("--name", required=True)
    p_train.add_argument("--audio", nargs="*", default=None)
    p_train.add_argument("--dataset-dir", default=None)
    p_train.add_argument("--metadata-csv", default=None)
    p_train.add_argument("--language", default="en")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--grad-accum", type=int, default=None)
    p_train.add_argument("--lr", type=float, default=5e-6)
    p_train.add_argument("--auto-transcribe", action="store_true")
    p_train.add_argument("--allow-cpu", action="store_true")
    p_train.add_argument("--allow-short", action="store_true")
    p_train.add_argument("--dry-run", action="store_true")

    # profiles
    p_prof = sub.add_parser("profiles", help="Manage saved user voice profiles.")
    p_prof.add_argument("action", choices=["list", "inspect", "delete"])
    p_prof.add_argument("name", nargs="?")

    # mcp
    sub.add_parser("mcp", help="Run the stdio JSON-RPC 2.0 MCP server.")

    return parser


def main(argv: _t.Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "mcp":
            from tts_app.mcp_server import main as mcp_main

            return mcp_main()

        if args.command == "status":
            payload = {
                "engines": [
                    {"id": e.id, "label": e.label, "supports_cloning": e.supports_cloning}
                    for e in available_engines()
                ],
                "hardware": check_hardware_requirements(),
                "voice_profiles": [p.to_dict() for p in list_profiles()],
            }
            if args.json:
                _print_json(payload)
            else:
                hw = payload["hardware"]
                print("Engines:", ", ".join(e["id"] for e in payload["engines"]))
                print(
                    f"Hardware: {hw['cpu_cores']} CPU cores | {hw['ram_total_gb']} GB RAM | "
                    f"{hw['disk_free_gb']} GB disk free | CUDA={hw['cuda_available']} "
                    f"{'(' + hw['gpu_name'] + ', ' + str(hw['vram_gb']) + ' GB VRAM)' if hw['gpu_name'] else ''}"
                )
                print(f"FFmpeg: {hw['ffmpeg_installed']} | Profiles: {len(payload['voice_profiles'])}")
                for tier_name, tier in hw["tiers"].items():
                    status = "READY" if tier["ready"] else "NOT READY"
                    print(f"  - {tier_name}: {status} ({tier['minimum']})")
                    for note in tier["notes"]:
                        print(f"      * {note}")
            return 0

        if args.command == "voices":
            engines = (
                available_engines()
                if args.engine.lower() in ("", "all")
                else [get_engine(args.engine)]
            )
            found = []
            for eng in engines:
                for v in eng.list_voices(args.language):
                    if args.gender and v.gender.lower() != args.gender.lower():
                        continue
                    if args.search and args.search.lower() not in (
                        f"{v.id} {v.name} {v.locale}".lower()
                    ):
                        continue
                    found.append(v.to_dict())
                    if len(found) >= args.limit:
                        break
                if len(found) >= args.limit:
                    break
            if args.json:
                _print_json({"voices": found})
            else:
                for v in found:
                    print(f"{v['engine']:8} | {v['locale']:8} | {v['gender']:7} | {v['id']} ({v['name']})")
            return 0

        if args.command == "speak":
            text = args.text
            if args.text_file:
                with open(args.text, "r", encoding="utf-8") as fh:
                    text = fh.read()
            ctx = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
            with ctx:
                res = smart_synthesize(
                    text=text,
                    output_path=os.path.abspath(args.output),
                    engine=args.engine,
                    language=args.language,
                    voice_id=args.voice,
                    gender=args.gender,
                    voice_profile=args.voice_profile,
                    reference_audio=args.reference_audio,
                    speed=args.speed,
                    pitch=args.pitch,
                    trim_silence_flag=args.trim,
                    normalize_loudness_flag=args.normalize,
                    generate_subtitles=args.subtitles,
                    fps=args.fps,
                    offline_only=args.offline,
                )
            if args.json:
                _print_json(res.to_dict())
            else:
                print(f"[{res.engine}] {res.voice_id} -> {res.audio_path} ({res.duration_s}s)")
                if res.srt_path:
                    print(f"Subtitles: {res.srt_path} | Timings: {res.timings_path}")
            return 0

        if args.command == "video-voiceover":
            script_text = args.script
            if args.script_file:
                with open(args.script_file, "r", encoding="utf-8") as fh:
                    script_text = fh.read()
            ctx = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
            with ctx:
                data = generate_video_voiceover(
                    script=script_text,
                    output_dir=args.output_dir,
                    output_name=args.name,
                    audio_format=args.format,
                    fps=args.fps,
                    default_engine=args.engine,
                    default_language=args.language,
                    default_voice_id=args.voice,
                    default_gender=args.gender,
                    default_voice_profile=args.voice_profile,
                    default_reference_audio=args.reference_audio,
                    default_speed=args.speed,
                    default_pause_ms=args.pause_ms,
                    trim_silence_flag=not args.no_trim,
                    normalize_loudness_flag=not args.no_normalize,
                    video_path=args.video,
                    burn_subtitles=args.burn_subtitles,
                )
            if args.json:
                _print_json(data)
            else:
                print(f"Master audio: {data['master_audio_path']} ({data['total_duration_s']}s / {data['total_frames']} frames)")
                print(f"Timeline JSON: {data['timeline_json_path']}")
                print(f"SRT: {data['srt_path']}")
                if data.get("muxed_video_path"):
                    print(f"Muxed video: {data['muxed_video_path']}")
            return 0

        if args.command == "enroll-voice":
            prof = enroll_voice_profile(
                name=args.name,
                audio_paths=args.audio,
                language=args.language,
                preprocess=not args.no_preprocess,
                notes=args.notes,
            )
            _print_json(prof.to_dict())
            return 0

        if args.command == "train-voice":
            res = train_custom_voice(
                profile_name=args.name,
                audio_paths=args.audio,
                dataset_dir=args.dataset_dir,
                metadata_csv=args.metadata_csv,
                language=args.language,
                epochs=args.epochs,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum,
                learning_rate=args.lr,
                auto_transcribe=args.auto_transcribe,
                allow_cpu=args.allow_cpu,
                allow_short=args.allow_short,
                dry_run=args.dry_run,
            )
            _print_json(res)
            return 0

        if args.command == "profiles":
            if args.action == "list":
                _print_json([p.to_dict() for p in list_profiles()])
            elif args.action == "inspect":
                if not args.name:
                    raise TtsError("Provide a profile name to inspect.")
                _print_json(get_profile(args.name).to_dict())
            elif args.action == "delete":
                if not args.name:
                    raise TtsError("Provide a profile name to delete.")
                _print_json({"deleted": delete_profile(args.name), "name": args.name})
            return 0
    except TtsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
