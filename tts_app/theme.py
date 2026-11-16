"""Modern light and dark QSS themes for the PyQt5 desktop studio."""

from __future__ import annotations

_PALETTES = {
    "dark": {
        "bg": "#0f172a",
        "surface": "#1e293b",
        "surface_alt": "#162032",
        "input": "#0b1220",
        "border": "#334155",
        "border_focus": "#6366f1",
        "text": "#f8fafc",
        "text_muted": "#94a3b8",
        "accent": "#6366f1",
        "accent_hover": "#4f46e5",
        "accent_pressed": "#4338ca",
        "danger": "#ef4444",
        "danger_hover": "#dc2626",
        "pill_bg": "#1e293b",
        "pill_text": "#38bdf8",
        "slider_track": "#334155",
        "slider_fill": "#6366f1",
        "check_bg": "#0b1220",
        "selection": "#312e81",
    },
    "light": {
        "bg": "#f1f5f9",
        "surface": "#ffffff",
        "surface_alt": "#f8fafc",
        "input": "#ffffff",
        "border": "#cbd5e1",
        "border_focus": "#4f46e5",
        "text": "#0f172a",
        "text_muted": "#475569",
        "accent": "#4f46e5",
        "accent_hover": "#4338ca",
        "accent_pressed": "#3730a3",
        "danger": "#dc2626",
        "danger_hover": "#b91c1c",
        "pill_bg": "#e0e7ff",
        "pill_text": "#3730a3",
        "slider_track": "#cbd5e1",
        "slider_fill": "#4f46e5",
        "check_bg": "#ffffff",
        "selection": "#e0e7ff",
    },
}


def build_stylesheet(dark: bool = True) -> str:
    """Return a clean, modern QSS stylesheet for dark or light mode."""
    p = _PALETTES["dark" if dark else "light"]
    return f"""
    QMainWindow, QDialog, QMessageBox {{
        background-color: {p['bg']};
        color: {p['text']};
        font-family: "Inter", "Segoe UI", "Ubuntu", "Noto Sans", sans-serif;
        font-size: 13px;
    }}
    QWidget {{
        background-color: transparent;
        color: {p['text']};
        font-family: "Inter", "Segoe UI", "Ubuntu", "Noto Sans", sans-serif;
        font-size: 13px;
    }}
    QWidget#AppRoot {{
        background-color: {p['bg']};
    }}
    QFrame#Card {{
        background-color: {p['surface']};
        border: 1px solid {p['border']};
        border-radius: 12px;
    }}
    QFrame#TransportBar {{
        background-color: {p['surface']};
        border-top: 1px solid {p['border']};
    }}
    QLabel#TitleLabel {{
        font-size: 19px;
        font-weight: 700;
        color: {p['text']};
    }}
    QLabel#SubtitleLabel {{
        font-size: 12px;
        color: {p['text_muted']};
    }}
    QLabel#SectionTitle {{
        font-size: 14px;
        font-weight: 600;
        color: {p['text']};
    }}
    QLabel#MutedLabel {{
        color: {p['text_muted']};
        font-size: 12px;
    }}
    QLabel#StatusBadge {{
        background-color: {p['pill_bg']};
        color: {p['pill_text']};
        border: 1px solid {p['border']};
        border-radius: 10px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 600;
    }}
    QTabWidget::pane {{
        border: none;
        margin-top: 8px;
    }}
    QTabBar::tab {{
        background-color: {p['surface']};
        color: {p['text_muted']};
        border: 1px solid {p['border']};
        border-radius: 8px;
        padding: 8px 20px;
        min-width: 130px;
        margin-right: 8px;
    }}
    QTabBar::tab:selected {{
        background-color: {p['accent']};
        color: #ffffff;
        border-color: {p['accent']};
    }}
    QTabBar::tab:hover:!selected {{
        color: {p['text']};
        border-color: {p['border_focus']};
    }}
    QPlainTextEdit, QLineEdit, QComboBox, QSpinBox, QListWidget {{
        background-color: {p['input']};
        color: {p['text']};
        border: 1px solid {p['border']};
        border-radius: 8px;
        padding: 6px 10px;
        min-height: 20px;
        selection-background-color: {p['selection']};
    }}
    QPlainTextEdit:focus, QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QListWidget:focus {{
        border: 1px solid {p['border_focus']};
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 26px;
        border: none;
    }}
    QComboBox QAbstractItemView {{
        background-color: {p['surface']};
        color: {p['text']};
        border: 1px solid {p['border']};
        selection-background-color: {p['accent']};
        selection-color: #ffffff;
        outline: none;
    }}
    QListWidget::item {{
        padding: 8px 10px;
        border-radius: 6px;
        margin: 2px 0px;
    }}
    QListWidget::item:selected {{
        background-color: {p['accent']};
        color: #ffffff;
    }}
    QPushButton {{
        background-color: {p['surface_alt']};
        color: {p['text']};
        border: 1px solid {p['border']};
        border-radius: 8px;
        padding: 7px 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        border-color: {p['border_focus']};
    }}
    QPushButton:disabled {{
        color: {p['text_muted']};
        opacity: 0.45;
    }}
    QPushButton#PrimaryButton {{
        background-color: {p['accent']};
        color: #ffffff;
        border: 1px solid {p['accent']};
    }}
    QPushButton#PrimaryButton:hover {{
        background-color: {p['accent_hover']};
        border-color: {p['accent_hover']};
    }}
    QPushButton#PrimaryButton:pressed {{
        background-color: {p['accent_pressed']};
    }}
    QPushButton#PrimaryButton:disabled {{
        background-color: {p['surface_alt']};
        color: {p['text_muted']};
        border-color: {p['border']};
    }}
    QPushButton#DangerButton {{
        background-color: transparent;
        color: {p['danger']};
        border: 1px solid {p['danger']};
    }}
    QPushButton#DangerButton:hover {{
        background-color: {p['danger']};
        color: #ffffff;
    }}
    QCheckBox {{
        spacing: 8px;
        color: {p['text']};
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: 1px solid {p['border']};
        background-color: {p['check_bg']};
    }}
    QCheckBox::indicator:hover {{
        border-color: {p['border_focus']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {p['accent']};
        border-color: {p['accent']};
    }}
    QSlider::groove:horizontal {{
        height: 6px;
        background: {p['slider_track']};
        border-radius: 3px;
    }}
    QSlider::sub-page:horizontal {{
        background: {p['slider_fill']};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: #ffffff;
        border: 2px solid {p['accent']};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 9px;
    }}
    QSplitter::handle {{
        background-color: transparent;
        width: 8px;
    }}
    QProgressBar {{
        background-color: {p['slider_track']};
        border: none;
        border-radius: 2px;
        height: 4px;
    }}
    QProgressBar::chunk {{
        background-color: {p['accent']};
        border-radius: 2px;
    }}
    """
