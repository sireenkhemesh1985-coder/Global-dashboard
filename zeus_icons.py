"""
ZEUS Global Market Intelligence — Centralized Enterprise Icon & Sanitization System.

Provides:
1. Monochromatic Lucide SVG icon helpers (16-18px, stroke-width 1.5-1.75px).
2. Material Symbol mappings for native Streamlit widgets where SVG HTML is unsupported.
3. Presentation-layer Unicode emoji sanitization to prevent decorative emojis in UI output.
"""

import re

# Palette constants aligned with Linear / Bloomberg / Vercel Enterprise design system
ICON_MUTED = "#94A3B8"
ICON_SUBTLE = "#64748B"
ICON_ACTIVE = "#38BDF8"
ICON_PRIMARY = "#F8FAFC"
ICON_SUCCESS = "#10B981"
ICON_WARNING = "#F59E0B"
ICON_DANGER = "#EF4444"

# Monochromatic Lucide SVG path definitions (24x24 viewBox)
_LUCIDE_PATHS = {
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "alert-triangle": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "arrow-up": '<path d="m18 15-6-6-6 6"/>',
    "award": '<circle cx="12" cy="8" r="6"/><path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/>',
    "bar-chart-2": '<line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/>',
    "book-open": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "briefcase": '<rect width="20" height="14" x="2" y="7" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
    "building-2": '<path d="M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18Z"/><path d="M6 12H4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2"/><path d="M18 9h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-2"/><path d="M10 6h4"/><path d="M10 10h4"/><path d="M10 14h4"/><path d="M10 18h4"/>',
    "check-circle-2": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "chevron-left": '<path d="m15 18-6-6 6-6"/>',
    "chevron-right": '<path d="m9 18 6-6-6-6"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "compass": '<circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "dollar-sign": '<line x1="12" x2="12" y1="2" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "lock": '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "log-out": '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" x2="9" y1="12" y2="12"/>',
    "mail": '<rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>',
    "map-pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    "scale": '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "shield-alert": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "sliders": '<line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/><line x1="18" x2="22" y1="16" y2="16"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "terminal": '<polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/>',
    "trending-up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
}

# Material Symbols mapping for native Streamlit widgets (st.button, st.expander, st.tabs, etc.)
MATERIAL_ICONS = {
    "market": ":material/public:",
    "strategy": ":material/shield:",
    "pricing": ":material/payments:",
    "outreach": ":material/mail:",
    "erp": ":material/analytics:",
    "terminal": ":material/terminal:",
    "video": ":material/play_circle:",
    "download": ":material/download:",
    "upload": ":material/upload:",
    "logout": ":material/logout:",
    "search": ":material/search:",
    "edit": ":material/edit:",
    "add": ":material/add:",
    "warning": ":material/warning:",
    "check": ":material/check_circle:",
    "document": ":material/description:",
    "settings": ":material/settings:",
    "compare": ":material/compare_arrows:",
    "tender": ":material/account_balance:",
}


def lucide_icon(
    name: str,
    size: int = 16,
    stroke_width: float = 1.65,
    color: str = ICON_MUTED,
    extra_style: str = "",
) -> str:
    """Return an inline monochromatic Lucide SVG icon (16-18px, 1.5-1.75px stroke)."""
    clamped_size = max(16, min(18, int(size)))
    clamped_stroke = max(1.5, min(1.75, float(stroke_width)))
    paths = _LUCIDE_PATHS.get(name, _LUCIDE_PATHS["activity"])
    style = f"flex-shrink:0;vertical-align:middle;{extra_style}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{clamped_size}" height="{clamped_size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{clamped_stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" style="{style}">{paths}</svg>'
    )


# Comprehensive regex covering Unicode emojis, regional flags, dingbats, variation selectors, ZWJ
_DECORATIVE_EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # Regional indicator flags
    "\U0001F300-\U0001F5FF"  # Misc symbols & pictographs
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F680-\U0001F6FF"  # Transport & map symbols
    "\U0001F700-\U0001F77F"  # Alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric shapes extended
    "\U0001F800-\U0001F8FF"  # Supplemental arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"  # Chess symbols
    "\U0001FA70-\U0001FAFF"  # Symbols & pictographs extended-A
    "\u2600-\u26FF"          # Miscellaneous symbols
    "\u2700-\u27BF"          # Dingbats
    "\u2300-\u23FF"          # Miscellaneous technical (⏱, ⌖, ⌬, ⎋)
    "\u2B00-\u2BFF"          # Misc symbols & arrows (⬆, ⭐, ⬡)
    "\u25C6\u25C7\u25C8\u2726\u229E"  # ◈, ◇, ◆, ✦, ⊞
    "\u2194-\u21AA\u2794"    # Decorative arrow symbols (↔, ↕, ↗, ↘, ➔)
    "\uFE0E\uFE0F\u200D"     # Variation selectors & ZWJ
    "]+"
)


def sanitize_ui_text(text):
    """Strip all decorative Unicode emojis from presentation-layer text while preserving data integrity."""
    if not isinstance(text, str) or not text:
        return text
    if not _DECORATIVE_EMOJI_RE.search(text):
        return text
    had_leading_emoji = bool(_DECORATIVE_EMOJI_RE.match(text))
    cleaned = _DECORATIVE_EMOJI_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    if had_leading_emoji:
        cleaned = cleaned.lstrip(" \t")
    return cleaned
