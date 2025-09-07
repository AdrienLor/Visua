# Visua

**Visua** is a fast, cross-platform image viewer built in Rust with a focus on speed, clarity, and efficient comparison workflows.

---

## Features

- **Dual-pane comparison mode (A/B)**
  - Linked **or independent** pan & zoom (toggle *Link* in the toolbar)  
  - Multiple comparison modes:  
    - **Split view** (side-by-side, with optional draggable divider)  
    - **Blink** (alternate A/B at fixed interval)  
    - **Overlay** (blend A & B with adjustable alpha)  
    - **Checkerboard** (tiled mix of A & B, adjustable tile size)  
    - **Diff** modes:  
      - **Gray** (continuous or thresholded)  
      - **Color** (per-channel |A−B|)  
      - **Heatmap** (false-color differences with adjustable gain)  
- **Supported formats**  
  - PNG, JPG, BMP, TIFF (incl. 16/32-bit float), WebP, GIF, TGA, ICO, HDR, PNM  
- **Sorting workflow**: quick triage of images into user-defined folders, with **on-the-fly folder creation** directly from the UI  
- **Image adjustments**: brightness, contrast, saturation, gamma  
- **Histogram view** (linear / logarithmic, per channel or luma)  
- **Navigation by folder** with independent browsing for A and B  
- **Transformations**: flip H/V, rotate 90° steps  
- **Slideshow mode** (fullscreen, distraction-free) with automatic playback and adjustable fade transition  
- **Modal dialogs** for *Properties* (including EXIF metadata) and *About*  
- **Lightweight, minimal UI** (built with [`egui`](https://github.com/emilk/egui) / [`eframe`](https://github.com/emilk/egui/tree/master/crates/eframe))  
- **Cross-platform**: Windows, Linux, macOS  

---

## HotFix (v1.4.1)

- Correct centering of images in **linked comparison mode** during rotation.  
- Fade-in effect now works consistently across modes (slideshow and normal view).  
- Rotation and mirroring logic cleaned up, with reset option available.  
- Removed unused `build_params_compare` path (simplified GPU pipeline).  

## What’s new (v1.5.0)

- **New comparison modes**:  
  - **Checkerboard** with adjustable tile size.  
  - Extended **Diff** with three visualization modes:  
    - Gray (continuous or thresholded).  
    - Color (per-channel differences).  
    - Heatmap (false-color map with adjustable gain).  
- **UI improvements**:  
  - Unified, mutually exclusive selector for Diff visualization modes.  
  - Context-sensitive controls (sliders/checks only when relevant).  

See the full [Changelog](CHANGELOG.md) for details.

---

## Roadmap

- Customizable keyboard shortcuts  
- Extended TIFF support (compression, metadata)  
- ICC color profile handling  
- ROI selection with quantitative metrics (MAE, PSNR, % above threshold)  
- Export current view (with adjustments and comparison mode applied)  
- Annotation and measurement tools  
- Performance optimizations for very large images  

---

## 📦 Build

```bash
cargo build --release
