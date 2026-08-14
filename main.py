"""Application entry point.

Run with ``python main.py`` from the project root (or ``python -m tts_app.gui``).
"""

import os
import sys


def _fix_frozen_gstreamer() -> None:
    """Help a PyInstaller-frozen app find the system GStreamer plugins.

    The QtMultimedia/GStreamer libraries bundled by PyInstaller can't locate
    the system plugin directory (e.g. /usr/lib/x86_64-linux-gnu/gstreamer-1.0),
    which makes every playback pipeline fail with ``no element "identity"``.
    Point the plugin system path at an existing directory before Qt starts.
    """
    if os.environ.get("GST_PLUGIN_SYSTEM_PATH"):
        return
    candidates = [
        "/usr/lib/x86_64-linux-gnu/gstreamer-1.0",
        "/usr/lib/aarch64-linux-gnu/gstreamer-1.0",
        "/usr/lib/arm-linux-gnueabihf/gstreamer-1.0",
        "/usr/lib64/gstreamer-1.0",
        "/usr/lib/gstreamer-1.0",
    ]
    for path in candidates:
        if os.path.isdir(path):
            os.environ["GST_PLUGIN_SYSTEM_PATH"] = path
            return


_fix_frozen_gstreamer()

from PyQt5.QtWidgets import QApplication, QMessageBox

from tts_app.gui import MainWindow
from tts_app.tts_engine import TtsError


def main() -> int:
    """Create the QApplication, show the window, and run the event loop."""
    app = QApplication(sys.argv)
    app.setApplicationName("Text-to-Speech")
    try:
        window = MainWindow()
    except TtsError as exc:
        QMessageBox.critical(None, "Startup error", str(exc))
        return 1
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
