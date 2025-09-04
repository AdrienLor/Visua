# Visua

**Visua** is a fast, cross-platform image viewer built in Rust with a focus on speed, clarity, and efficient comparison workflows.

---

## Features

- **Dual-pane comparison mode (A/B)**
  - Linked **or independent** pan & zoom (toggle *Link* in the toolbar)  
  - Split view, blink mode, or pixel-wise difference mode  
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

## HotFixe (v.1.4.1)

- Correct centering of images in **linked comparison mode** during rotation.  
- Fade-in effect now works consistently across modes (slideshow and normal view).  
- Rotation and mirroring logic cleaned up, with reset option available.  
- Removed unused `build_params_compare` path (simplified GPU pipeline).  

## What’s new (v1.4.0)

- **Smooth fade-in transitions** when loading images  
- **Slideshow improvements**:  
  - Adjustable fade duration (0 to ~6s)  
  - Navigation with arrows, mouse click, or wheel  
- **Rotation by 90° increments** (replaces old fixed 180° toggle)  
- **Better centering & fit logic** with rotated images  
- **Modal windows** for properties and about panel, blocking background interactions  
- **Improved comparison tools**: blink and diff modes polished, with proper centering and alpha handling  
- **Minor UI refinements

See the full [Changelog](CHANGELOG.md) for details.

---

## Roadmap

- Customizable keyboard shortcuts  
- Extended TIFF support (compression, metadata)  
- ICC color profile handling  
- Annotation and measurement tools  
- Performance optimizations for very large images  

---

## 📦 Build

```bash
cargo build --release
