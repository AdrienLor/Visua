# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- …

### Changed
- …

### Fixed
- …

---

## [1.1.0] – 2025-08-21

### Added
- 🎞 **Slideshow mode**:
  - Fullscreen, distraction-free display of image A.
  - Manual navigation with ← / → keys, exit with Esc.
  - Toggle with F11 or toolbar button.
- ⏱ **Automatic slideshow playback** with adjustable timer (1–30 seconds).
- ℹ **About dialog** with software name, version, author, and supported formats.
- Bottom **status bar** for temporary success/error messages on white background.

### Changed
- Sorting workflow:
  - No default subfolder is preselected.
  - If no subfolders exist, the user must create one before sorting.
- UI layout refinements for cleaner toolbar and dialogs.

### Fixed
- Window initialization now respects centering and minimum size rules.
- Prevented images from being re-saved into the parent folder when no subfolder was chosen.

---

## [1.0.0] – 2025-08-15

### Added
- Initial public release of **Visua**.
- Dual-pane comparison mode (A/B) with synchronized zoom & pan.
- Support for common formats:
  - PNG, JPG, BMP, TIFF (incl. 32-bit float), WebP, GIF, TGA, ICO, HDR, PNM.
- Support for scientific and document formats:
  - FITS (astronomy, pure Rust reader).
  - PDF (via Pdfium integration).
- Basic image adjustments: brightness, contrast, saturation, gamma.
- Histogram view (linear / log).
- Transformations: flip, mirror, rotate 180°.
- Clean minimal UI with egui/eframe.
