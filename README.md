# Visua

**Visua** is a fast, cross-platform image viewer built in Rust with a focus on speed, clarity, and efficient comparison workflows.

---

## ✨ Features

- **Dual-pane comparison mode (A/B)**
  - Linked **or independent** pan & zoom (toggle *Link* in the toolbar)  
  - Side-by-side or overlay comparison  
- **Supported formats**  
  - PNG, JPG, BMP, TIFF (incl. 16/32-bit float), WebP, GIF, TGA, ICO, HDR, PNM
- **Sorting workflow**: quick triage of images into user-defined folders, with **on-the-fly folder creation** directly from the UI  
- **Image adjustments**: brightness, contrast, saturation, gamma  
- **Histogram view** (linear / logarithmic)  
- **Navigation by folder** with independent browsing for A and B  
- **Transformations**: flip H/V, rotate 180°  
- **Slideshow mode** (fullscreen, distraction-free) with automatic playback  
- **Lightweight, minimal UI** (built with [`egui`](https://github.com/emilk/egui) / [`eframe`](https://github.com/emilk/egui/tree/master/crates/eframe))  
- **Cross-platform**: Windows, Linux, macOS  

---

## 🆕 What’s new (v1.3.0)
- Independent pan & zoom in split comparison mode (Link toggle)   
- Asynchronous image loading (no flicker, old image stays visible until swap)  
- Improved navigation: path bars update instantly for both panes   

See the full [Changelog](CHANGELOG.md) for details.

---

## 🚀 Roadmap
- Customizable keyboard shortcuts  
- Extended TIFF handling (compression, metadata)  
- Annotation and measurement tools  
- Color profiles / ICC support  

---

## 📦 Build
```bash
cargo build --release
