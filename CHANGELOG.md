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

## [1.3.0] – 2025-08-31

### Added
- 🔓 **Independent pan & zoom in split comparison mode**:  
  - New **Link toggle** in the toolbar (linked by default).  
  - When unlinked, each pane (A/B) can be panned/zoomed independently with mouse drag & scroll.  
- ➖ **Split divider**:  
  - Optional vertical divider between A and B.  
  - Toggleable from the toolbar.    

### Changed
- ⚡ **Asynchronous image loading**:  
  - Old image remains visible until the new one is ready (no flicker).  
  - Navigation and folder loads update immediately in the status bar.  
- Navigation wrappers (`navigate_a` / `navigate_b`) now update paths immediately, avoiding stale info in the secondary bar.  
- Fit / Center / 1:1 logic unified with proper handling of linked vs. independent split modes.  

### Fixed
- ✅ **No more UI flicker** when navigating between images (textures are swapped atomically on load).  
- ✅ Paths in the secondary info bar (B) now update correctly on folder and image change.  
- ✅ 1:1 zoom now behaves correctly in both linked and independent split modes.  
- Improved repaint handling: no unnecessary full redraws, smoother updates when loading.  

---

## [1.2.0] – 2025-08-24

### Changed
- 🧹 **Simplified core**: removed support for PDF and FITS formats to keep Visua lightweight and focused on image viewing.  
- Image loading pipeline now centered on standard formats + advanced TIFF handling (incl. 16/32-bit float).  
- Updated README and documentation to reflect the new scope.  

---

## [1.1.0] – 2025-08-21

### Added
- 🎞 **Slideshow mode** (fullscreen, distraction-free, image A only).  
  - Navigation:  
    - → Arrow / Left Click / Scroll Up → Next image  
    - ← Arrow / Scroll Down → Previous image  
    - Esc / Right Click → Exit slideshow  
  - Toggle with **F11** or 🎞 toolbar button.  
- ⏱ **Automatic slideshow playback** with configurable interval (1–30s).  
- ℹ **About dialog (modal)** with integrated slideshow help and blocked background interactions.  
- 📂 **New folder dialog (modal)** with proper validation and blocked background interactions.  
- Bottom **status bar** for temporary messages (success/error).  

### Changed
- Toolbar and dialogs refined for clarity.  
- Diaporama and About are now **true modal dialogs**: background dimming + no interaction possible behind.  
- Slideshow defaults to fullscreen without window borders.  

### Fixed
- Prevented images from being accidentally re-saved in the parent folder when no subfolder was selected.  
- Window centering logic now applies consistently on startup.  
- Focus handling improved in modal dialogs (no loss of interaction if clicking outside).  

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
- Minimal, clean UI with egui/eframe.  

---

[Unreleased]: https://github.com/your-org/visua/compare/v1.3.0...HEAD  
[1.3.0]: https://github.com/your-org/visua/compare/v1.2.0...v1.3.0  
[1.2.0]: https://github.com/your-org/visua/compare/v1.1.0...v1.2.0  
[1.1.0]: https://github.com/your-org/visua/compare/v1.0.0...v1.1.0  
[1.0.0]: https://github.com/your-org/visua/releases/tag/v1.0.0
