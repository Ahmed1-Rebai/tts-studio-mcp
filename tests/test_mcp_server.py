"""End-to-end stdio JSON-RPC 2.0 tests for the MCP server."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest


def _rpc_roundtrip(requests: list[dict], env_extra: dict[str, str] | None = None) -> list[dict]:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    payload = "\n".join(json.dumps(r) for r in requests) + "\n"
    proc = subprocess.run(
        [sys.executable, "-m", "tts_app.mcp_server"],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    replies = []
    for line in proc.stdout.splitlines():
        if line.strip():
            replies.append(json.loads(line))
    return replies


def test_mcp_stdio_protocol_and_tools(tmp_path):
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 4, "method": "resources/list"},
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/read",
            "params": {"uri": "tts://guides/hardware-and-training"},
        },
        {"jsonrpc": "2.0", "id": 6, "method": "prompts/list"},
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "tts_status", "arguments": {}},
        },
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {"name": "manage_voice_profiles", "arguments": {"action": "list"}},
        },
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "unknown_tool", "arguments": {}},
        },
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "tts_status", "arguments": ["not-an-object"]},
        },
    ]
    replies = _rpc_roundtrip(requests, env_extra={"TTS_APP_DATA_DIR": str(tmp_path)})
    by_id = {r["id"]: r for r in replies}

    # Notification produced no reply (10 replies for 11 requests).
    assert len(replies) == 10
    assert by_id[1]["result"]["serverInfo"]["name"] == "tts-mcp"
    assert by_id[2]["result"] == {}
    tool_names = {t["name"] for t in by_id[3]["result"]["tools"]}
    assert {
        "tts_status",
        "list_voices",
        "synthesize_speech",
        "generate_video_voiceover",
        "mux_audio_to_video",
        "enroll_voice_profile",
        "train_custom_voice",
        "manage_voice_profiles",
    }.issubset(tool_names)

    assert "Minimum Hardware Tiers" in by_id[5]["result"]["contents"][0]["text"]
    assert by_id[7]["result"]["isError"] is False
    assert "hardware" in by_id[7]["result"]["structuredContent"]
    assert by_id[8]["result"]["structuredContent"]["profiles"] == []
    assert by_id[9]["result"]["isError"] is True
    assert by_id[10]["result"]["isError"] is True


def test_mcp_synthesize_speech_keeps_stdout_clean(tmp_path):
    if importlib.util.find_spec("pyttsx3") is None:
        pytest.skip("pyttsx3 not installed")

    out_wav = str(tmp_path / "mcp_clip.wav")
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "synthesize_speech",
                "arguments": {
                    "text": "MCP voiceover integration test.",
                    "output_path": out_wav,
                    "format": "wav",
                    "engine": "pyttsx3",
                    "generate_subtitles": True,
                },
            },
        },
    ]
    replies = _rpc_roundtrip(requests, env_extra={"TTS_APP_DATA_DIR": str(tmp_path)})
    assert len(replies) == 2
    res = replies[1]["result"]
    assert res["isError"] is False
    sc = res["structuredContent"]
    assert os.path.isfile(sc["audio_path"])
    assert os.path.isfile(sc["srt_path"])
    assert os.path.isfile(sc["timings_path"])
