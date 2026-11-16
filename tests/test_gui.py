"""Offscreen Qt UI tests for the modern Voice Studio desktop GUI."""

from __future__ import annotations

import os

import pytest


def test_modern_gui_tabs_theme_and_voice_search(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("TTS_APP_DATA_DIR", str(tmp_path))

    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    from tts_app.gui import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    win = MainWindow()
    try:
        assert win._tabs.count() == 3
        assert win._tabs.tabText(0) == "Speech Studio"
        assert win._tabs.tabText(1) == "Video Voiceover"
        assert win._tabs.tabText(2) == "Voice Profiles"

        # Theme toggle switches dark <-> light cleanly
        assert win._dark_mode is True
        win.on_toggle_theme()
        assert win._dark_mode is False
        win.on_toggle_theme()
        assert win._dark_mode is True

        # Word stats & speed reset
        win._text_edit.setPlainText("Hello world from Voice Studio.")
        assert "5 words" in win._word_stats.text()
        win._speed_slider.setValue(125)
        assert win._speed_label.text() == "1.25x"
        win._btn_reset_speed.click()
        assert win._speed_label.text() == "1.00x"

        # Copy script to Video Voiceover tab
        win._on_copy_to_video_tab()
        assert win._video_script_edit.toPlainText() == "Hello world from Voice Studio."

        # Switch through available engines & exercise voice search on catalog engine
        for idx in range(win._engine_combo.count()):
            win._engine_combo.setCurrentIndex(idx)
            if win.current_engine().has_voice_catalog:
                total = win._voice_combo.count()
                win._voice_search.setText("zzzz_nonexistent_voice_query")
                assert win._voice_combo.count() == 0
                win._voice_search.clear()
                assert win._voice_combo.count() == total
    finally:
        win.close()
