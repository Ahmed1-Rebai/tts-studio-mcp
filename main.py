"""Application entry point.

Run with ``python main.py`` from the project root (or ``python -m tts_app.gui``).
"""

import sys

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
