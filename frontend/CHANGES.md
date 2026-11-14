# What changed
- New `refine.css` (linked last in index.html). No JS, ids or onclick handlers were touched.
- Fonts: Inter/JetBrains Mono -> IBM Plex Sans/Mono. Palette: cool grey-teal -> warm paper + signal yellow.
- Dark ink sidebar in both themes; flat 2px corners; no shadows; severity tags as stamped squares; status dots as square block signals.
- Segmented controls became underlined tab strips; KPI numbers set in mono.
- Fake "AI Detection Accuracy 99.4%" replaced by "False alarms per camera-hour 0.6 (limit 1.0)", matching NFR-02 in the SRS.
- Bug fixed: `#view-overview{display:block !important}` kept the Overview visible under every other tab. Now `.active` only.
- Light/dark toggle (existing) kept and verified; theme still saved in localStorage.
- `index.original.html` is the untouched version for comparison.
