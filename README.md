# Visua

**Visua** is a cross-platform image viewer built in Rust with a focus on speed and clarity.

## ✨ Features
- Dual-pane comparison mode (A/B) with synchronized zoom & pan
- Support for common image formats: PNG, JPG, BMP, TIFF (incl. 16/32-bit float), WebP, GIF, TGA, ICO, HDR, PNM
- Basic image adjustments: brightness, contrast, saturation, gamma
- Histogram view (linear / logarithmic)
- Navigation by folder with independent browsing for A and B
- Transformation tools: flip (H/V), mirror
- Lightweight and minimal UI (built with `egui` / `eframe`)
- Cross-platform (Windows, Linux, macOS)

## 🚀 Roadmap
- Customizable keyboard shortcuts
- Extended TIFF handling 
- Annotation and measurement tools

## 📦 Build
```bash
cargo build --release
