#![windows_subsystem = "windows"]
#![allow(dead_code)]

// uses
use egui::{IconData, RichText};
use bytemuck::{Pod, Zeroable};
use eframe::{
    egui,
    egui::{Pos2, Rect, Sense, UiBuilder, Vec2},
    egui_wgpu,
};
use eframe::egui_wgpu::{Callback as WgpuCallback, wgpu};
use rfd::FileDialog;
use std::{
    fs,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    sync::mpsc::{channel, Sender, Receiver},
    thread::{self, JoinHandle},
    time::{Instant, Duration},
    collections::HashMap,
};
use num_cpus;
use exif;

fn install_panic_hook_once() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            let _ = std::fs::write(
                "crash.log",
                format!("panic: {info}\nbacktrace:\n{:?}\n",
                        std::backtrace::Backtrace::force_capture()),
            );
        }));
    });
}

//=================ASYNC=====================
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ReqId(u64);

#[derive(Clone, Copy, Debug)]
enum Target {
    MainPaneA,
    MainPaneB,
}

enum Job {
    Load {
        id: ReqId,
        path: PathBuf,
        target: Target,
        max_side: u32,
    },
    Quit,
}

enum JobResult {
    Ok {
        id: ReqId,
        target: Target,
        size: [usize; 2],
        rgba: Arc<Vec<u8>>, 
    },
    Err {
        id: ReqId,
        target: Target,
        error: String,
    },
}

struct Loader {
    tx: Sender<Job>,
    rx: Receiver<JobResult>,
    workers: Vec<JoinHandle<()>>,
}

impl Drop for Loader {
    fn drop(&mut self) {
        // Envoyer autant de Quit que de workers
        for _ in 0..self.workers.len() {
            let _ = self.tx.send(Job::Quit);
        }
        // Fermer l’émetteur pour débloquer d’éventuels recv
        let _ = &self.tx;
        // Rejoindre les threads
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

fn start_loader(num_threads: usize) -> Loader {
    let (tx_job, rx_job) = channel::<Job>();
    let (tx_res, rx_res) = channel::<JobResult>();

    let shared_rx = Arc::new(Mutex::new(rx_job));
    let n = num_threads.max(1);

    let mut workers = Vec::with_capacity(n);
    for _ in 0..n {
        let rx = Arc::clone(&shared_rx);
        let tx_res_cl = tx_res.clone();
        workers.push(thread::spawn(move || {
            loop {
                let job = {
                    let lock = rx.lock().unwrap();
                    // Si le sender est fermé => break
                    match lock.recv() {
                        Ok(j) => j,
                        Err(_) => break,
                    }
                };
                match job {
                    Job::Load { id, path, target, max_side } => {
                        let res = (|| -> Result<(Vec<u8>, [usize;2]), String> {
                            let (mut data, dims) = load_any_rgba8(&path)?;
                            let (w, h) = (dims[0] as u32, dims[1] as u32);
                            let max0 = w.max(h);
                            if max0 > max_side {
                                let img = image::RgbaImage::from_raw(w, h, std::mem::take(&mut data))
                                    .ok_or_else(|| "buffer RGBA invalide".to_string())?;
                                let scale = max_side as f32 / max0 as f32;
                                let nw = ((w as f32 * scale).round() as u32).max(1);
                                let nh = ((h as f32 * scale).round() as u32).max(1);
                                let resized = image::DynamicImage::ImageRgba8(img)
                                    .resize_exact(nw, nh, image::imageops::FilterType::Triangle)
                                    .to_rgba8();
                                let (nw, nh) = resized.dimensions();
                                Ok((resized.into_raw(), [nw as usize, nh as usize]))
                            } else {
                                Ok((data, [w as usize, h as usize]))
                            }
                        })();

                        match res {
                            Ok((rgba, size)) => {
                                let _ = tx_res_cl.send(JobResult::Ok {
                                    id, target, size, rgba: Arc::new(rgba)
                                });
                            }
                            Err(e) => {
                                let _ = tx_res_cl.send(JobResult::Err { id, target, error: e });
                            }
                        }
                    }
                    Job::Quit => break,
                }
            }
        }));
    }

    Loader { tx: tx_job, rx: rx_res, workers }
}

//==============================================

fn is_supported_ext(p: &Path) -> bool {
    let ext = p.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    matches!(ext.as_str(),
        "jpg"|"jpeg"|"png"|"bmp"|"gif"|"webp"|"tif"|"tiff")
}

#[inline]
fn scissor_from_info(info: &egui::PaintCallbackInfo) -> Option<(u32, u32, u32, u32)> {
    // Rect de clip “demandé” par egui (i32)
    let cr = info.clip_rect_in_pixels();
    // Viewport de la passe (framebuffer cible), i32
    let vp = info.viewport_in_pixels();

    // Bords du viewport
    let vx1 = vp.left_px.max(0);
    let vy1 = vp.top_px.max(0);
    let vx2 = vx1 + vp.width_px.max(0);
    let vy2 = vy1 + vp.height_px.max(0);

    // Bords du clip
    let cx1 = cr.left_px;
    let cy1 = cr.top_px;
    let cx2 = cr.left_px + cr.width_px;
    let cy2 = cr.top_px + cr.height_px;

    // Intersection clip ∩ viewport
    let ix1 = cx1.max(vx1);
    let iy1 = cy1.max(vy1);
    let ix2 = cx2.min(vx2);
    let iy2 = cy2.min(vy2);

    let iw = (ix2 - ix1).max(0);
    let ih = (iy2 - iy1).max(0);
    if iw == 0 || ih == 0 {
        return None; // zone vide -> ne rien dessiner
    }
    Some((ix1 as u32, iy1 as u32, iw as u32, ih as u32))
}

// ---- Budgets mémoire / tailles maxi ----
const MAX_SIDE: u32 = 20_000;                 // côté max accepté
const MAX_PIXELS_SOFT: u64 = 120_000_000;     // ~120 MP → on downscale
const MAX_PIXELS_HARD: u64 = 220_000_000;     // >220 MP → on refuse
const MAX_GPU_BYTES: u64 = 512 * 1024 * 1024; // ~512 MiB par texture
const MAX_TEX_SIDE_FALLBACK: u32 = 8192;

#[inline]
fn estimate_cost(w: u32, h: u32) -> (u64, u64, u64) {
    let cpu = 4u64 * w as u64 * h as u64;
    let row = ((4u64 * w as u64) + 255) & !255; // align 256 B
    let gpu = row * h as u64;
    (cpu, gpu, cpu + gpu)
}

// Lecture rapide des dimensions TIFF (sans décoder les pixels)
fn peek_tiff_dims(path: &std::path::Path) -> Result<(u32, u32), String> {
    use std::fs::File;
    use std::io::BufReader;
    use tiff::decoder::Decoder;
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut dec = Decoder::new(BufReader::new(file)).map_err(|e| e.to_string())?;
    let (w, h) = dec.dimensions().map_err(|e| e.to_string())?;
    Ok((w, h))
}

// Fallback générique (image crate) pour JPG/PNG/etc.
fn peek_generic_dims(path: &std::path::Path) -> Result<(u32, u32), String> {
    image::image_dimensions(path).map_err(|e| e.to_string())
}

// Redimensionnement d’un buffer RGBA8
fn resize_rgba8_buf(rgba: Vec<u8>, w: u32, h: u32, tw: u32, th: u32) -> Result<Vec<u8>, String> {
    use image::{DynamicImage, ImageBuffer, Rgba};
    let img = ImageBuffer::<Rgba<u8>, _>::from_vec(w, h, rgba)
        .ok_or_else(|| "buffer invalide pour ImageBuffer".to_string())?;
    let dynimg = DynamicImage::ImageRgba8(img)
        .resize_exact(tw, th, image::imageops::FilterType::Lanczos3);
    Ok(dynimg.to_rgba8().into_raw())
}

// ======================= Conversions =======================

fn normalize_min_max_f32(buf: &[f32]) -> (f32, f32) {
    let mut minv = f32::INFINITY;
    let mut maxv = f32::NEG_INFINITY;
    for &v in buf {
        if v.is_finite() {
            if v < minv {
                minv = v;
            }
            if v > maxv {
                maxv = v;
            }
        }
    }
    if !minv.is_finite() || !maxv.is_finite() { (0.0, 1.0) }
    else { (minv, maxv) 
    }
}

fn map_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn rgba_from_gray_u8(gray: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 4];
    for i in 0..(w * h) {
        let g = gray[i];
        out[4 * i] = g;
        out[4 * i + 1] = g;
        out[4 * i + 2] = g;
        out[4 * i + 3] = 255;
    }
    out
}

fn rgba_from_gray_u16(gray: &[u16], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 4];
    for i in 0..(w * h) {
        let u = map_to_u8(gray[i] as f32 / 65535.0);
        out[4 * i] = u;
        out[4 * i + 1] = u;
        out[4 * i + 2] = u;
        out[4 * i + 3] = 255;
    }
    out
}

fn rgba_from_gray_f32(gray: &[f32], w: usize, h: usize) -> Vec<u8> {
    let (mn, mx) = normalize_min_max_f32(gray);
    let denom = (mx - mn).abs();
    let scale = if denom > 1e-20 { 1.0 / denom } else { 1.0 }; // image constante
    let mut out = vec![0u8; w * h * 4];
    for i in 0..(w * h) {
        let u = map_to_u8((gray[i] - mn) * scale);
        out[4 * i] = u;
        out[4 * i + 1] = u;
        out[4 * i + 2] = u;
        out[4 * i + 3] = 255;
    }
    out
}

fn rgba_from_rgb_u8(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 4];
    for i in 0..(w * h) {
        out[4 * i] = rgb[3 * i];
        out[4 * i + 1] = rgb[3 * i + 1];
        out[4 * i + 2] = rgb[3 * i + 2];
        out[4 * i + 3] = 255;
    }
    out
}

fn rgba_from_rgb_u16(rgb: &[u16], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 4];
    for i in 0..(w * h) {
        out[4 * i] = map_to_u8(rgb[3 * i] as f32 / 65535.0);
        out[4 * i + 1] = map_to_u8(rgb[3 * i + 1] as f32 / 65535.0);
        out[4 * i + 2] = map_to_u8(rgb[3 * i + 2] as f32 / 65535.0);
        out[4 * i + 3] = 255;
    }
    out
}

fn rgba_from_rgb_f32(rgb: &[f32], w: usize, h: usize) -> Vec<u8> {
    let (mn, mx) = normalize_min_max_f32(rgb);
    let denom = (mx - mn).abs();
    let scale = if denom > 1e-20 { 1.0 / denom } else { 1.0 }; // image constante
    let mut out = vec![0u8; w * h * 4];
    for i in 0..(w * h) {
        out[4 * i] = map_to_u8((rgb[3 * i] - mn) * scale);
        out[4 * i + 1] = map_to_u8((rgb[3 * i + 1] - mn) * scale);
        out[4 * i + 2] = map_to_u8((rgb[3 * i + 2] - mn) * scale);
        out[4 * i + 3] = 255;
    }
    out
}

// ======================= TIFF =======================

fn load_tiff_rgba8(path: &Path) -> Result<(Vec<u8>, [usize; 2]), String> {
    use tiff::decoder::{Decoder, DecodingResult};
    use tiff::ColorType;

    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut dec = Decoder::new(BufReader::new(file)).map_err(|e| e.to_string())?;
    let (w, h) = dec.dimensions().map_err(|e| e.to_string())?;
    let ct = dec.colortype().map_err(|e| e.to_string())?;
    let img = dec.read_image().map_err(|e| e.to_string())?;

    let wh = [w as usize, h as usize];

    let rgba = match (ct, img) {
        (ColorType::Gray(_), DecodingResult::U8(v)) => rgba_from_gray_u8(&v, wh[0], wh[1]),
        (ColorType::Gray(_), DecodingResult::U16(v)) => rgba_from_gray_u16(&v, wh[0], wh[1]),
        (ColorType::Gray(_), DecodingResult::F32(v)) => rgba_from_gray_f32(&v, wh[0], wh[1]),
        (ColorType::RGB(_), DecodingResult::U8(v)) => rgba_from_rgb_u8(&v, wh[0], wh[1]),
        (ColorType::RGB(_), DecodingResult::U16(v)) => rgba_from_rgb_u16(&v, wh[0], wh[1]),
        (ColorType::RGB(_), DecodingResult::F32(v)) => rgba_from_rgb_f32(&v, wh[0], wh[1]),
        (ColorType::RGBA(_), DecodingResult::U8(v)) => v,
        (ColorType::RGBA(_), DecodingResult::U16(v)) => {
            let mut out = vec![0u8; wh[0] * wh[1] * 4];
            for i in 0..(wh[0] * wh[1]) {
                out[4 * i] = map_to_u8(v[4 * i] as f32 / 65535.0);
                out[4 * i + 1] = map_to_u8(v[4 * i + 1] as f32 / 65535.0);
                out[4 * i + 2] = map_to_u8(v[4 * i + 2] as f32 / 65535.0);
                out[4 * i + 3] = map_to_u8(v[4 * i + 3] as f32 / 65535.0);
            }
            out
        }
        (ColorType::RGBA(_), DecodingResult::F32(v)) => {
            let (mn, mx) = normalize_min_max_f32(&v);
            let denom = (mx - mn).abs();
            let scale = if denom > 1e-20 { 1.0 / denom } else { 1.0 }; // image constante
            let mut out = vec![0u8; wh[0] * wh[1] * 4];
            for i in 0..(wh[0] * wh[1]) {
                out[4 * i] = map_to_u8((v[4 * i] - mn) * scale);
                out[4 * i + 1] = map_to_u8((v[4 * i + 1] - mn) * scale);
                out[4 * i + 2] = map_to_u8((v[4 * i + 2] - mn) * scale);
                out[4 * i + 3] = map_to_u8((v[4 * i + 3] - mn) * scale);
            }
            out
        }
        _ => {
            let dynimg = image::open(path)
                .map_err(|e| e.to_string())?
                .to_rgba8();
            let (w, h) = dynimg.dimensions();
            let wh2 = [w as usize, h as usize];
            return Ok((dynimg.into_raw(), wh2));
        }
    };

    Ok((rgba, wh))
}


fn parse_val_num(s: &str) -> Option<f64> {
    let t = s
        .trim()
        .trim_matches('\'')
        .replace('D', "E")
        .replace('d', "E");
    t.parse::<f64>().ok()
}

//====================================== EXIF =========================================

fn extract_exif(path: &std::path::Path) -> Result<Vec<(String, String)>, String> {
   
    let file = File::open(path).map_err(|e| format!("open: {e}"))?;
    let mut bufreader = BufReader::new(&file);

    let exifreader = exif::Reader::new();
    let exif = exifreader
        .read_from_container(&mut bufreader)
        .map_err(|e| format!("parse: {e}"))?;

    let mut seen = HashMap::new();

    for f in exif.fields() {
        let label = match f.tag {
            exif::Tag::Make                     => Some("Camera Make"),
            exif::Tag::Model                    => Some("Camera Model"),
            exif::Tag::Software                 => Some("Software"),
            exif::Tag::DateTimeOriginal         => Some("Date Taken"),
            exif::Tag::ExposureTime             => Some("Exposure Time"),
            exif::Tag::FNumber                  => Some("Aperture (f-stop)"),
            exif::Tag::PhotographicSensitivity  => Some("ISO"),
            exif::Tag::FocalLength              => Some("Focal Length"),
            exif::Tag::ExposureBiasValue        => Some("Exposure Bias"),
            exif::Tag::Flash                    => Some("Flash"),
            exif::Tag::WhiteBalance             => Some("White Balance"),
            exif::Tag::MeteringMode             => Some("Metering Mode"),
            exif::Tag::Orientation              => Some("Orientation"),
            exif::Tag::XResolution              => Some("X Resolution (DPI)"),
            exif::Tag::YResolution              => Some("Y Resolution (DPI)"),
            exif::Tag::ResolutionUnit           => Some("Resolution Unit"),
            _ => None,
        };

        if let Some(name) = label {
            let raw = f.display_value().with_unit(&exif).to_string();

            // Formattage et traductions
            let value = match f.tag {
                exif::Tag::ExposureTime => {
                    match &f.value {
                        exif::Value::Rational(vec) if !vec.is_empty() => {
                            let r = vec[0];
                            if r.num > 0 {
                                let denom = r.denom as f64 / r.num as f64;
                                format!("1/{:.0} s", denom.round())
                            } else {
                                raw
                            }
                        }
                        _ => raw,
                    }
                }
                exif::Tag::FNumber => {
                    match &f.value {
                        exif::Value::Rational(vec) if !vec.is_empty() => {
                            let r = vec[0];
                            let val = r.num as f64 / r.denom as f64;
                            format!("f/{:.1}", val)
                        }
                        _ => raw,
                    }
                }
                exif::Tag::PhotographicSensitivity => {
                    match f.value.get_uint(0) {
                        Some(v) => format!("ISO {}", v),
                        None => raw,
                    }
                }
                exif::Tag::FocalLength => {
                    match &f.value {
                        exif::Value::Rational(vec) if !vec.is_empty() => {
                            let r = vec[0];
                            let val = r.num as f64 / r.denom as f64;
                            format!("{:.1} mm", val)
                        }
                        _ => raw,
                    }
                }
                exif::Tag::ExposureBiasValue => {
                    match &f.value {
                        exif::Value::SRational(vec) if !vec.is_empty() => {
                            let r = vec[0];
                            let val = r.num as f64 / r.denom as f64;
                            if val > 0.0 {
                                format!("+{:.1} EV", val)
                            } else {
                                format!("{:.1} EV", val)
                            }
                        }
                        _ => raw,
                    }
                }
                exif::Tag::Flash => match raw.to_lowercase().as_str() {
                    s if s.contains("not fired") => "Not fired".to_string(),
                    s if s.contains("fired") => "Fired".to_string(),
                    _ => raw,
                },
                exif::Tag::WhiteBalance => match raw.to_lowercase().as_str() {
                    s if s.contains("auto") => "Auto".to_string(),
                    s if s.contains("manual") => "Manual".to_string(),
                    _ => raw,
                },
                exif::Tag::Orientation => match raw.as_str() {
                    s if s.contains("row 0 at top and column 0 at left") => "Top-left (normal)".to_string(),
                    s if s.contains("row 0 at top and column 0 at right") => "Top-right (mirrored)".to_string(),
                    s if s.contains("row 0 at bottom and column 0 at right") => "Bottom-right (rotated 180°)".to_string(),
                    s if s.contains("row 0 at bottom and column 0 at left") => "Bottom-left (mirrored, rotated 180°)".to_string(),
                    s if s.contains("row 0 at left and column 0 at top") => "Left-top (rotated 90° CCW)".to_string(),
                    s if s.contains("row 0 at right and column 0 at top") => "Right-top (rotated 90° CW)".to_string(),
                    s if s.contains("row 0 at right and column 0 at bottom") => "Right-bottom (mirrored, rotated 90° CW)".to_string(),
                    s if s.contains("row 0 at left and column 0 at bottom") => "Left-bottom (mirrored, rotated 90° CCW)".to_string(),
                    _ => raw,
                },
                _ => raw,
            };

            // Déduplique : 1 tag = 1 ligne
            seen.entry(f.tag)
                .or_insert_with(|| (name.to_string(), value));
        }
    }

    if seen.is_empty() {
        return Err("No relevant EXIF metadata found".into());
    }

    // --- Ordre logique fixe ---
    let order = [
        "Camera Make",
        "Camera Model",
        "Software",
        "Date Taken",
        "Exposure Time",
        "Aperture (f-stop)",
        "ISO",
        "Focal Length",
        "Exposure Bias",
        "Flash",
        "White Balance",
        "Metering Mode",
        "Orientation",
        "X Resolution (DPI)",
        "Y Resolution (DPI)",
        "Resolution Unit",
    ];

    let mut out = Vec::new();
    for key in order {
        if let Some((_k, v)) = seen.values().find(|(name, _)| name == key) {
            out.push((key.to_string(), v.clone()));
        }
    }

    Ok(out)
}

fn start_meta_loader() -> MetaLoader {
    let (tx_job, rx_job) = channel::<MetaJob>();
    let (tx_res, rx) = channel::<MetaResult>();

    std::thread::spawn(move || {
        while let Ok(job) = rx_job.recv() {
            let res = match extract_exif(&job.path) {
                Ok(items) => MetaResult::Ok { pane: job.pane, items },
                Err(e)    => MetaResult::Err { pane: job.pane, error: e },
            };
            let _ = tx_res.send(res);
        }
    });

    MetaLoader { tx: tx_job, rx }
}

#[derive(Clone, Copy)]
enum Pane { A, B }

struct MetaJob {
    pane: Pane,
    path: PathBuf,
}

enum MetaResult {
    Ok  { pane: Pane, items: Vec<(String, String)> },
    Err { pane: Pane, error: String },
}

struct MetaLoader {
    tx: Sender<MetaJob>,
    rx: Receiver<MetaResult>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PropsTab { Properties, Exif }

fn build_properties_for(size: [usize; 2], path: &std::path::Path) -> Vec<(String, String)> {
    let mut out = Vec::<(String,String)>::new();

    // Path / name / extension
    out.push(("Path".into(), path.display().to_string()));
    out.push(("File name".into(), path.file_name().unwrap_or_default().to_string_lossy().to_string()));
    out.push(("Extension".into(), path.extension().unwrap_or_default().to_string_lossy().to_string()));

    // File size + modified date
    if let Ok(md) = std::fs::metadata(path) {
        let bytes = md.len();
        let human = if bytes >= 1<<20 { format!("{:.2} MiB", bytes as f64 / (1<<20) as f64) }
                    else if bytes >= 1<<10 { format!("{:.2} KiB", bytes as f64 / (1<<10) as f64) }
                    else { format!("{bytes} B") };
        out.push(("File size".into(), human));

        if let Ok(modified) = md.modified() {
            let dt: chrono::DateTime<chrono::Local> = modified.into();
            out.push(("Modified".into(), dt.format("%Y-%m-%d %H:%M:%S").to_string()));
        }
    }

    // Dimensions
    if size != [0,0] {
        let (w,h) = (size[0], size[1]);
        out.push(("Dimensions".into(), format!("{w} × {h} px")));
        out.push(("Pixels".into(), format!("{:.3} MPix", (w as f64 * h as f64)/1_000_000.0)));
    } else {
        out.push(("Dimensions".into(), "n/a".into()));
        out.push(("Pixels".into(), "n/a".into()));
    }

    out.push(("Color model".into(), "n/a".into()));
    out.push(("DPI".into(), "n/a".into()));
    out.push(("Compression".into(), "n/a".into()));
    out.push(("Chroma subsampling".into(), "n/a".into()));
    out.push(("Color profile".into(), "n/a".into()));

    out
}

fn ui_props_table(
    ui: &mut egui::Ui,
    props_a: &[(String, String)],
    props_b: &[(String, String)],
    dual: bool,
) {
    use egui_extras::{TableBuilder, Column};

    // ordre fixe
    let order = [
        "Path",
        "File name",
        "Extension",
        "File size",
        "Modified",
        "Dimensions",
        "Pixels",
        "Color model",
        "DPI",
        "Compression",
        "Chroma subsampling",
        "Color profile",
    ];

    if dual {
        TableBuilder::new(ui)
            .striped(true)
            .column(Column::auto().resizable(false).at_least(100.0))
            .column(Column::remainder())
            .column(Column::remainder())
            .body(|mut body| {
                for key in order {
                    let val_a = props_a.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str()).unwrap_or("—");
                    let val_b = props_b.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str()).unwrap_or("—");

                    // on saute la ligne si A et B == "n/a"
                    if val_a == "n/a" && val_b == "n/a" {
                        continue;
                    }

                    body.row(20.0, |mut row| {
                        row.col(|ui| { let _ = ui.monospace(key); });
                        row.col(|ui| {
                            if val_a == "n/a" {
                                let _ = ui.weak("n/a");
                            } else {
                                let _ = ui.monospace(val_a);
                            }
                        });
                        row.col(|ui| {
                            if val_b == "n/a" {
                                let _ = ui.weak("n/a");
                            } else {
                                let _ = ui.monospace(val_b);
                            }
                        });
                    });
                }
            });
    } else {
        TableBuilder::new(ui)
            .striped(true)
            .column(Column::auto().resizable(false).at_least(100.0))
            .column(Column::remainder())
            .body(|mut body| {
                for key in order {
                    if let Some((_k, v)) = props_a.iter().find(|(k, _)| k == key) {
                        if v == "n/a" {
                            continue; // on saute les champs inutiles
                        }
                        body.row(20.0, |mut row| {
                            row.col(|ui| { let _ = ui.monospace(key); });
                            row.col(|ui| { let _ = ui.monospace(v); });
                        });
                    }
                }
            });
    }
}

fn ui_props_exif(
    ui: &mut egui::Ui,
    meta_a: &[(String, String)],
    meta_b: &[(String, String)],
    dual: bool,
    inflight_a: bool,
    inflight_b: bool,
    err_a: Option<&str>,
    err_b: Option<&str>,
) {
    use egui_extras::{TableBuilder, Column};

    // Bandeau d'état
    ui.horizontal(|ui| {
        if inflight_a {
            let _ = ui.label("A: parsing…");
        } else if let Some(err) = err_a {
            let _ = ui.colored_label(egui::Color32::RED, format!("A: {err}"));
        } else {
            let _ = ui.weak(format!("A: {} tags", meta_a.len()));
        }

        if dual {
            ui.separator();
            if inflight_b {
                let _ = ui.label("B: parsing…");
            } else if let Some(err) = err_b {
                let _ = ui.colored_label(egui::Color32::RED, format!("B: {err}"));
            } else {
                let _ = ui.weak(format!("B: {} tags", meta_b.len()));
            }
        }
    });
    ui.separator();

    if dual {
        TableBuilder::new(ui)
            .striped(true)
            .column(Column::auto().resizable(false).at_least(140.0)) // Tag
            .column(Column::remainder())            // Valeur A
            .column(Column::remainder())            // Valeur B
            .body(|mut body| {
                for (key, val_a) in meta_a {
                    // chercher la valeur correspondante dans B
                    let val_b = meta_b.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str()).unwrap_or("n/a");

                    // ignorer si A et B = "n/a"
                    if val_a == "n/a" && val_b == "n/a" {
                        continue;
                    }

                    body.row(20.0, |mut row| {
                        row.col(|ui| { let _ = ui.monospace(key); });

                        row.col(|ui| {
                            if val_a == "n/a" {
                                let _ = ui.weak("n/a");
                            } else {
                                let _ = ui.monospace(val_a);
                            }
                        });

                        row.col(|ui| {
                            if val_b == "n/a" {
                                let _ = ui.weak("n/a");
                            } else {
                                let _ = ui.monospace(val_b);
                            }
                        });
                    });
                }
            });
    } else {
        TableBuilder::new(ui)
            .striped(true)
            .column(Column::auto().resizable(false).at_least(140.0)) // Tag
            .column(Column::remainder())            // Valeur A
            .body(|mut body| {
                for (key, val_a) in meta_a {
                    if val_a == "n/a" {
                        continue; // ignorer complètement
                    }
                    body.row(20.0, |mut row| {
                        row.col(|ui| { let _ = ui.monospace(key); });
                        row.col(|ui| { let _ = ui.monospace(val_a); });
                    });
                }
            });
    }
}

// ======================= Dispatcher formats =======================

fn supported_ext(ext: &str) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "png" | "jpg" | "jpeg" | "bmp" | "tiff" | "tif" | "webp" | "gif" | "tga" | "ico" | "pnm" | "hdr"
    )
}

fn load_any_rgba8(path: &Path) -> Result<(Vec<u8>, [usize; 2]), String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    if matches!(ext.as_str(), "tif" | "tiff") {
        return load_tiff_rgba8(path);
    }

    // Formats gérés par `image` (jpg/png/webp/…)
    let img = image::ImageReader::open(path)
        .map_err(|e| e.to_string())?
        .with_guessed_format()
        .map_err(|e| e.to_string())?
        .decode()
        .map_err(|e| e.to_string())?;

    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Ok((rgba.into_raw(), [w as usize, h as usize]))
}

// ======================= Histogrammes =======================

#[derive(Clone)]
struct HistOut {
    luma: [u32; 256],
    r: [u32; 256],
    g: [u32; 256],
    b: [u32; 256],
}

#[inline]
fn compute_hist_or_zero(
    rgba_opt: Option<&Arc<Vec<u8>>>,
    size: [usize; 2],
    brightness: f32,
    contrast: f32,
    saturation: f32,
    gamma: f32,
    sample_target: usize,
) -> ([u32;256],[u32;256],[u32;256],[u32;256]) {
    if let Some(buf) = rgba_opt {
        let out = compute_hist_rgba(
            buf,
            size,
            brightness,
            contrast,
            saturation,
            gamma,
            sample_target,
        );
        (out.luma, out.r, out.g, out.b)
    } else {
        ([0;256],[0;256],[0;256],[0;256])
    }
}

fn compute_hist_rgba(
    rgba: &[u8],
    size: [usize; 2],
    brightness: f32,
    contrast: f32,
    saturation: f32,
    gamma: f32,
    sample_target: usize,
) -> HistOut {
    let (w, h) = (size[0], size[1]);

    // Taille max réellement lisible d'après le buffer
    let max_px_from_buf = rgba.len() / 4;
    if max_px_from_buf == 0 {
        return HistOut { luma: [0;256], r: [0;256], g: [0;256], b: [0;256] };
    }

    // Bornage: ne jamais lire au-delà du buffer, même si (w*h) est incohérent
    let total_px = w.saturating_mul(h);
    let n = total_px.min(max_px_from_buf);

    let stride = if n > sample_target && sample_target > 0 {
        (n + sample_target - 1) / sample_target
    } else {
        1
    };

    let mut out = HistOut { luma: [0;256], r: [0;256], g: [0;256], b: [0;256] };

    let mut idx = 0usize;
    while idx < n {
        let i4 = idx * 4;
        let mut rf = rgba[i4] as f32 / 255.0;
        let mut gf = rgba[i4 + 1] as f32 / 255.0;
        let mut bf = rgba[i4 + 2] as f32 / 255.0;

        // Luminosité/contraste
        rf = ((rf + brightness).clamp(0.0, 1.0) - 0.5) * contrast + 0.5;
        gf = ((gf + brightness).clamp(0.0, 1.0) - 0.5) * contrast + 0.5;
        bf = ((bf + brightness).clamp(0.0, 1.0) - 0.5) * contrast + 0.5;

        // Saturation
        let l = 0.2126 * rf + 0.7152 * gf + 0.0722 * bf;
        rf = (l + (rf - l) * saturation).clamp(0.0, 1.0);
        gf = (l + (gf - l) * saturation).clamp(0.0, 1.0);
        bf = (l + (bf - l) * saturation).clamp(0.0, 1.0);

        // Gamma
        if (gamma - 1.0).abs() > f32::EPSILON {
            rf = rf.powf(gamma);
            gf = gf.powf(gamma);
            bf = bf.powf(gamma);
        }

        let ir = (rf * 255.0).round().clamp(0.0, 255.0) as usize;
        let ig = (gf * 255.0).round().clamp(0.0, 255.0) as usize;
        let ib = (bf * 255.0).round().clamp(0.0, 255.0) as usize;
        let il = (0.2126 * rf + 0.7152 * gf + 0.0722 * bf) * 255.0;
        let il = il.round().clamp(0.0, 255.0) as usize;

        out.r[ir] += 1;
        out.g[ig] += 1;
        out.b[ib] += 1;
        out.luma[il] += 1;

        idx += stride;
    }

    out
}

fn draw_histogram_luma(ui: &mut egui::Ui, hist: &[u32; 256], height: f32, log_scale: bool) {
    let width = 256.0;
    let (rect, _resp) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
    let max = hist.iter().copied().max().unwrap_or(1) as f32;

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, egui::Color32::from_gray(30));

    let log_max = (1.0 + max).ln();
    for (i, &v) in hist.iter().enumerate() {
        let value = v as f32;
        let h = if max > 0.0 {
            if log_scale {
                ((1.0 + value).ln() / log_max) * (height - 2.0)
            } else {
                (value / max) * (height - 2.0)
            }
        } else {
            0.0
        };

        let x = rect.left() + i as f32;
        let y = rect.bottom() - h;
        let bar = Rect::from_min_max(Pos2::new(x, y), Pos2::new(x + 1.0, rect.bottom()));
        painter.rect_filled(bar, 0.0, egui::Color32::from_gray(220));
    }
    painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, egui::Color32::GRAY));
}

fn draw_histogram_rgb(
    ui: &mut egui::Ui,
    r: &[u32; 256],
    g: &[u32; 256],
    b: &[u32; 256],
    height: f32,
    log_scale: bool,
) {
    let width = 256.0;
    let (rect, _resp) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
    let max = r
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .max(g.iter().copied().max().unwrap_or(1))
        .max(b.iter().copied().max().unwrap_or(1)) as f32;

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, egui::Color32::from_gray(30));

    let log_max = (1.0 + max).ln();
    let height2 = height - 2.0;

    let draw_channel = |p: &egui::Painter, data: &[u32; 256], color: egui::Color32| {
        for (i, &v) in data.iter().enumerate() {
            let value = v as f32;
            let h = if max > 0.0 {
                if log_scale {
                    ((1.0 + value).ln() / log_max) * height2
                } else {
                    (value / max) * height2
                }
            } else {
                0.0
            };
            let x = rect.left() + i as f32;
            let y = rect.bottom() - h;
            let bar = Rect::from_min_max(Pos2::new(x, y), Pos2::new(x + 1.0, rect.bottom()));
            p.rect_filled(bar, 0.0, color);
        }
    };

    let red = egui::Color32::from_rgba_unmultiplied(255, 64, 64, 180);
    let green = egui::Color32::from_rgba_unmultiplied(64, 255, 64, 180);
    let blue = egui::Color32::from_rgba_unmultiplied(64, 128, 255, 180);

    draw_channel(&painter, r, red);
    draw_channel(&painter, g, green);
    draw_channel(&painter, b, blue);

    painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, egui::Color32::GRAY));
}

// ======================= GPU (WGSL post-process) =======================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    brightness: f32,
    contrast:   f32,
    saturation: f32,
    gamma:      f32,
    flip_h:     u32,
    flip_v:     u32,
    rotation:   u32,
    _pad0:      u32,
    zoom:       f32,
    _pad1:      f32,
    tex_w:      f32,
    tex_h:      f32,
    off_x:      f32,
    off_y:      f32,
    center_u:   f32,
    center_v:   f32,
    rect_min_x: f32, rect_min_y: f32,
    rect_max_x: f32, rect_max_y: f32,
    ppp:        f32,
    fade_alpha: f32,
    _pad2:      [f32; 3],   // garde l’alignement (vec3 côté WGSL)
    _pad3:      [f32; 4],   // <-- NOUVEAU : pousse la taille totale à 112 octets
}

const POST_WGSL: &str = r#"
struct Params {
    brightness: f32,
    contrast:   f32,
    saturation: f32,
    gamma:      f32,
    flip_h:     u32,
    flip_v:     u32,
    rotation:   u32,
    _pad0:      u32,
    zoom:       f32,
    _pad1:      f32,
    tex_w:      f32,
    tex_h:      f32,
    off_x:      f32,
    off_y:      f32,
    center_u:   f32,
    center_v:   f32,
    rect_min_x: f32, rect_min_y: f32,
    rect_max_x: f32, rect_max_y: f32,
    ppp:        f32,
    fade_alpha: f32,
    _pad2:      vec3<f32>,
};
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> P: Params;

struct VSOut { @builtin(position) pos: vec4<f32>, };

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    // Fullscreen triangle (couvre tout l'écran sans dépendre du viewport)
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var o: VSOut;
    o.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    return o;
}

fn clamp01(v: vec3<f32>) -> vec3<f32> { return clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)); }

fn apply_bcs_gamma(rgb: vec3<f32>) -> vec3<f32> {
    var c = rgb + vec3<f32>(P.brightness);
    c = (c - vec3<f32>(0.5)) * vec3<f32>(P.contrast) + vec3<f32>(0.5);
    let l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    c = mix(vec3<f32>(l), c, vec3<f32>(P.saturation));
    c = clamp01(c);
    c = pow(c, vec3<f32>(max(P.gamma, 1e-6)));
    return clamp01(c);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    // pixels -> points
    let fx = frag_pos.x / P.ppp;
    let fy = frag_pos.y / P.ppp;

    // repère local au rectangle de dessin (en points)
    let rx = fx - P.rect_min_x;
    let ry = fy - P.rect_min_y;
    let rw = max(P.rect_max_x - P.rect_min_x, 1e-6);
    let rh = max(P.rect_max_y - P.rect_min_y, 1e-6);

    var tw: f32;
    var th: f32;
    if (P.rotation == 90u || P.rotation == 270u) {
        tw = max(P.tex_h * P.zoom, 1e-6);
        th = max(P.tex_w * P.zoom, 1e-6);
    } else {
        tw = max(P.tex_w * P.zoom, 1e-6);
        th = max(P.tex_h * P.zoom, 1e-6);
    }

    // mapping UV (non clampé)
    var uu: f32;
    var vv: f32;

    if (P.center_u >= 0.0) {
        let cx = rw * 0.5 + P.off_x;
        let cy = rh * 0.5 + P.off_y;
        let tlx = cx - P.center_u * tw;
        let tly = cy - P.center_v * th;
        uu = (rx - tlx) / tw;
        vv = (ry - tly) / th;
    } else {
        uu = (rx - P.off_x) / tw;
        vv = (ry - P.off_y) / th;
    }

    // flips / rotation
    if (P.flip_h == 1u) { uu = 1.0 - uu; }
    if (P.flip_v == 1u) { vv = 1.0 - vv; }

    // rotation selon l'angle
    if (P.rotation == 90u) {
        let tmp = uu;
        uu = vv;
        vv = 1.0 - tmp;
    } else if (P.rotation == 180u) {
        uu = 1.0 - uu;
        vv = 1.0 - vv;
    } else if (P.rotation == 270u) {
        let tmp = uu;
        uu = 1.0 - vv;
        vv = tmp;
    }

    // Si en dehors de [0,1], rendre TRANSPARENT (montre le fond)
    let inside = (uu >= 0.0) && (uu <= 1.0) && (vv >= 0.0) && (vv <= 1.0);
    if (!inside) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);   // alpha 0 => fond visible
        // Variante: `discard;` si tu préfères couper net (performant mais pas “blendé”)
    }

    // Sampling avec UV clampés (pour éviter les artefacts aux bords)
    let u = clamp(uu, 0.0, 1.0);
    let v = clamp(vv, 0.0, 1.0);

    let col = textureSample(tex, samp, vec2<f32>(u, v));
    let out_rgb = apply_bcs_gamma(col.rgb);

    // 🎬 fade-in simple : multiplie la sortie par alpha
    return vec4<f32>(out_rgb, 1.0) * P.fade_alpha;
}
"#;

const POST_WGSL_DIFF: &str = r#"
struct Params {
    brightness: f32,
    contrast:   f32,
    saturation: f32,
    gamma:      f32,
    flip_h:     u32,
    flip_v:     u32,
    rotation:   u32,
    _pad0:      u32,
    zoom:       f32,
    _pad1:      f32,
    tex_w:      f32,
    tex_h:      f32,
    off_x:      f32,
    off_y:      f32,
    center_u:   f32,
    center_v:   f32,
    rect_min_x: f32, rect_min_y: f32,
    rect_max_x: f32, rect_max_y: f32,
    ppp:        f32,
    fade_alpha: f32,
    _pad2:      vec3<f32>,
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var texA: texture_2d<f32>;
@group(0) @binding(2) var texB: texture_2d<f32>;
@group(0) @binding(3) var<uniform> P: Params;

struct VSOut { @builtin(position) pos: vec4<f32>, };

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    // Fullscreen triangle
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var o: VSOut;
    o.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    return o;
}

fn clamp01(v: vec3<f32>) -> vec3<f32> { return clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)); }
fn apply_bcs_gamma(rgb: vec3<f32>) -> vec3<f32> {
    var c = rgb + vec3<f32>(P.brightness);
    c = (c - vec3<f32>(0.5)) * vec3<f32>(P.contrast) + vec3<f32>(0.5);
    let l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    c = mix(vec3<f32>(l), c, vec3<f32>(P.saturation));
    c = clamp01(c);
    c = pow(c, vec3<f32>(max(P.gamma, 1e-6)));
    return clamp01(c);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let fx = frag_pos.x / P.ppp;
    let fy = frag_pos.y / P.ppp;

    let rx = fx - P.rect_min_x;
    let ry = fy - P.rect_min_y;
    let rw = max(P.rect_max_x - P.rect_min_x, 1e-6);
    let rh = max(P.rect_max_y - P.rect_min_y, 1e-6);

    var tw: f32;
    var th: f32;
    if (P.rotation == 90u || P.rotation == 270u) {
        tw = max(P.tex_h * P.zoom, 1e-6);
        th = max(P.tex_w * P.zoom, 1e-6);
    } else {
        tw = max(P.tex_w * P.zoom, 1e-6);
        th = max(P.tex_h * P.zoom, 1e-6);
    }

    var u: f32;
    var v: f32;

    if (P.center_u >= 0.0) {
        let cx = rw * 0.5 + P.off_x;
        let cy = rh * 0.5 + P.off_y;
        let tlx = cx - P.center_u * tw;
        let tly = cy - P.center_v * th;
        u = (rx - tlx) / tw;
        v = (ry - tly) / th;
    } else {
        u = (rx - P.off_x) / tw;
        v = (ry - P.off_y) / th;
    }

    if (P.flip_h == 1u) { u = 1.0 - u; }
    if (P.flip_v == 1u) { v = 1.0 - v; }

    // rotation selon l'angle
    if (P.rotation == 90u) {
        let tmp = u;
        u = v;
        v = 1.0 - tmp;
    } else if (P.rotation == 180u) {
        u = 1.0 - u;
        v = 1.0 - v;
    } else if (P.rotation == 270u) {
        let tmp = u;
        u = 1.0 - v;
        v = tmp;
    }

    // Transparence hors image pour montrer le fond
    if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let colA = textureSample(texA, samp, vec2<f32>(u, v)).rgb;
    let colB = textureSample(texB, samp, vec2<f32>(u, v)).rgb;

    var diff = abs(colA - colB);
    diff = clamp01(diff * vec3<f32>(1.0));
    let out_rgb = apply_bcs_gamma(diff);
    return vec4<f32>(out_rgb, 1.0);
}
"#;

// Ressources GPU par draw (créées en prepare, utilisées en paint)
struct GpuStuff {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    ubo: wgpu::Buffer,
    _texture: wgpu::Texture,
}

struct PostProcessCb {
    pixels: Arc<Vec<u8>>,
    size: [u32; 2],
    params: GpuParams,
    linear: bool,
    gpu: Mutex<Option<GpuStuff>>,
}

impl PostProcessCb {
    fn new(pixels: Arc<Vec<u8>>, size: [usize; 2], params: GpuParams, linear: bool) -> Self {
        Self {
            pixels,
            size: [size[0] as u32, size[1] as u32],
            params,
            linear,
            gpu: Mutex::new(None),
        }
    }
}

struct GpuStuff2 {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    ubo: wgpu::Buffer,
    _tex_a: wgpu::Texture,
    _tex_b: wgpu::Texture,
}

struct PostProcessCbDiff {
    pixels_a: Arc<Vec<u8>>,
    size_a: [u32; 2],
    pixels_b: Arc<Vec<u8>>,
    size_b: [u32; 2],
    params: GpuParams,
    linear: bool,
    gpu: Mutex<Option<GpuStuff2>>,
}

impl PostProcessCbDiff {
    fn new(pa: Arc<Vec<u8>>, sa: [usize; 2], pb: Arc<Vec<u8>>, sb: [usize; 2], params: GpuParams, linear: bool) -> Self {
        Self {
            pixels_a: pa,
            size_a: [sa[0] as u32, sa[1] as u32],
            pixels_b: pb,
            size_b: [sb[0] as u32, sb[1] as u32],
            params,
            linear,
            gpu: Mutex::new(None),
        }
    }
}

impl egui_wgpu::CallbackTrait for PostProcessCbDiff {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_desc: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        _resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut guard = self.gpu.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            // textures A et B
            let make_tex = |size: [u32;2]| -> wgpu::Texture {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("uv_diff_tex"),
                    size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm, // non-sRGB
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                })
            };
            let tex_a = make_tex(self.size_a);
            let tex_b = make_tex(self.size_b);

            // upload
            queue.write_texture(
                wgpu::ImageCopyTexture { texture: &tex_a, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &self.pixels_a,
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4 * self.size_a[0]), rows_per_image: Some(self.size_a[1]) },
                wgpu::Extent3d { width: self.size_a[0], height: self.size_a[1], depth_or_array_layers: 1 },
            );
            queue.write_texture(
                wgpu::ImageCopyTexture { texture: &tex_b, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &self.pixels_b,
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4 * self.size_b[0]), rows_per_image: Some(self.size_b[1]) },
                wgpu::Extent3d { width: self.size_b[0], height: self.size_b[1], depth_or_array_layers: 1 },
            );

            let view_a = tex_a.create_view(&wgpu::TextureViewDescriptor::default());
            let view_b = tex_b.create_view(&wgpu::TextureViewDescriptor::default());

            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("uv_sampler_diff"),
                mag_filter: if self.linear { wgpu::FilterMode::Linear } else { wgpu::FilterMode::Nearest },
                min_filter: if self.linear { wgpu::FilterMode::Linear } else { wgpu::FilterMode::Nearest },
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("post_shader_diff"),
                source: wgpu::ShaderSource::Wgsl(POST_WGSL_DIFF.into()),
            });

            // BGL: samp, texA, texB, UBO
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("post_bgl_diff"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<GpuParams>() as u64),
                        },
                        count: None,
                    },
                ],
            });

            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("post_pl_diff"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

            let target_format = wgpu::TextureFormat::Bgra8Unorm; // doit matcher egui
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("post_pipeline_diff"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            let ubo = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("post_ubo_diff"),
                size: std::mem::size_of::<GpuParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post_bind_diff"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Sampler(&sampler) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view_a) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&view_b) },
                    wgpu::BindGroupEntry { binding: 3, resource: ubo.as_entire_binding() },
                ],
            });

            *guard = Some(GpuStuff2 { pipeline, bind_group: bind, ubo, _tex_a: tex_a, _tex_b: tex_b });
        }

        if let Some(gpu) = guard.as_ref() {
            let mut p = self.params;
            p.ppp = screen_desc.pixels_per_point;
            queue.write_buffer(&gpu.ubo, 0, bytemuck::bytes_of(&p));
        }
        Vec::new()
    }

    fn paint(&self, info: egui::PaintCallbackInfo, rpass: &mut wgpu::RenderPass<'static>, _resources: &egui_wgpu::CallbackResources) {
        if let Some(gpu) = self.gpu.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            let Some((left, top, width, height)) = scissor_from_info(&info) else {
                return; // hors viewport ou zone nulle
            };
            rpass.set_scissor_rect(left, top, width, height);

            rpass.set_pipeline(&gpu.pipeline);
            rpass.set_bind_group(0, &gpu.bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
    }
}

impl egui_wgpu::CallbackTrait for PostProcessCb {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_desc: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        _resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut guard = self.gpu.lock().unwrap_or_else(|e| e.into_inner());

        if guard.is_none() {
            // Texture RGBA8 sRGB depuis CPU
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("uv_image_tex"),
                size: wgpu::Extent3d {
                    width: self.size[0],
                    height: self.size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &self.pixels,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * self.size[0]),
                    rows_per_image: Some(self.size[1]),
                },
                wgpu::Extent3d {
                    width: self.size[0],
                    height: self.size[1],
                    depth_or_array_layers: 1,
                },
            );
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("uv_sampler"),
                mag_filter: if self.linear {
                    wgpu::FilterMode::Linear
                } else {
                    wgpu::FilterMode::Nearest
                },
                min_filter: if self.linear {
                    wgpu::FilterMode::Linear
                } else {
                    wgpu::FilterMode::Nearest
                },
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("post_shader"),
                source: wgpu::ShaderSource::Wgsl(POST_WGSL.into()),
            });

            // BGL
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("post_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(
                                std::mem::size_of::<GpuParams>() as u64
                            ),
                        },
                        count: None,
                    },
                ],
            });

            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("post_pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

            // IMPORTANT : doit matcher le format du RenderPass (sur ta config: Bgra8Unorm)
            let target_format = wgpu::TextureFormat::Bgra8Unorm;

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("post_pipeline"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            // UBO
            let ubo = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("post_ubo"),
                size: std::mem::size_of::<GpuParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post_bind"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ubo.as_entire_binding(),
                    },
                ],
            });

            *guard = Some(GpuStuff {
                pipeline,
                bind_group: bind,
                ubo,
                _texture: tex,
            });
        }

        if let Some(gpu) = guard.as_ref() {
            let mut p = self.params;
            // Conversion points<->pixels : laisser au shader, on envoie juste ppp ici
            p.ppp = screen_desc.pixels_per_point;
            queue.write_buffer(&gpu.ubo, 0, bytemuck::bytes_of(&p));
        }

        Vec::new()
    }

fn paint(
    &self,
    info: egui::PaintCallbackInfo,
    rpass: &mut wgpu::RenderPass<'static>,
    _resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        if let Some(gpu) = self.gpu.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        let Some((left, top, width, height)) = scissor_from_info(&info) else {
            return; // hors viewport ou zone nulle
        };
        rpass.set_scissor_rect(left, top, width, height);

        rpass.set_pipeline(&gpu.pipeline);
        rpass.set_bind_group(0, &gpu.bind_group, &[]);
        rpass.draw(0..3, 0..1);
        }
    }
}

fn make_postprocess_paint_callback(
    rect: egui::Rect,
    pixels: Arc<Vec<u8>>,
    size: [usize; 2],
    linear: bool,
    mut params: GpuParams,
) -> egui::PaintCallback {
    params.rect_min_x = rect.min.x;
    params.rect_min_y = rect.min.y;
    params.rect_max_x = rect.max.x;
    params.rect_max_y = rect.max.y;
    let cb = PostProcessCb::new(pixels, size, params, linear);
    WgpuCallback::new_paint_callback(rect, cb)
}

fn make_postprocess_paint_callback_diff(
    rect: egui::Rect,
    px_a: Arc<Vec<u8>>, size_a: [usize; 2],
    px_b: Arc<Vec<u8>>, size_b: [usize; 2],
    linear: bool,
    mut params: GpuParams,
) -> egui::PaintCallback {
    params.rect_min_x = rect.min.x;
    params.rect_min_y = rect.min.y;
    params.rect_max_x = rect.max.x;
    params.rect_max_y = rect.max.y;
    let cb = PostProcessCbDiff::new(px_a, size_a, px_b, size_b, params, linear);
    WgpuCallback::new_paint_callback(rect, cb)
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum CompareMode {
    Split,
    Blink,
    Diff,  
}

impl CompareMode {
    #[inline]
    fn label(self) -> &'static str {
        match self {
            CompareMode::Split => "Side/Side",
            CompareMode::Blink => "Blink",
            CompareMode::Diff  => "Diff",
            // CompareMode::Overlay => "Overlay",      
            // CompareMode::Swipe   => "Swipe",        
        }
    }
}

// ======================= App =======================

struct App {
    // A
    orig_a: Option<Arc<Vec<u8>>>,
    size_a: [usize; 2],
    tex_a_cpu: Option<egui::TextureHandle>,

    // B
    orig_b: Option<Arc<Vec<u8>>>,
    size_b: [usize; 2],
    tex_b_cpu: Option<egui::TextureHandle>,

     // --- Async loader ---
    loader: Loader,
    next_req_id: u64,
    last_req_a: Option<ReqId>,
    last_req_b: Option<ReqId>,
    inflight_a: bool,
    inflight_b: bool,

    // mode slideshow
    slideshow_mode: bool,
    auto_slideshow: bool,
    slideshow_interval: f32,    // en secondes
    slideshow_timer: f32,       // compteur interne

    // Drop différé des textures egui (pour éviter Queue::submit sur texture détruite)
    pending_free: Vec<egui::TextureHandle>,

    // Navigation
    filelist_a: Vec<PathBuf>,
    idx_a: usize,
    filelist_b: Vec<PathBuf>,
    idx_b: usize,

    // Panneau options
    show_options: bool,

    //fenetre about
    show_about: bool,

    // Vue
    zoom: f32,
    min_zoom: f32,
    max_zoom: f32,
    offset: Vec2,
    request_fit: bool,
    request_center: bool,
    request_one_to_one: bool,
    keep_center_on_resize: bool,
    auto_fit_on_transform: bool,
    last_panel_size: Option<Vec2>,

    fit_allow_upscale: bool,

    // Zoom step (% / cran)
    zoom_step_percent: f32,
   
    // Module de tri
    subfolders: Vec<String>,       // liste dynamique
    show_new_folder_dialog: bool,  // afficher le popup ?
    new_folder_input: String,      // saisie utilisateur

    // Transfos
    rotation : u32,
    flip_h: bool,
    flip_v: bool,

    // Rendu
    linear_filter: bool,
    bg_gray: u8, 

    // Ajustements
    brightness: f32,
    contrast: f32,
    saturation: f32,
    gamma: f32,

    // Histogrammes (A)
    hist_luma: [u32; 256],
    hist_r: [u32; 256],
    hist_g: [u32; 256],
    hist_b: [u32; 256],
    hist_dirty: bool,
    log_hist: bool,
    hist_rgb_mode: bool,

    // Histogrammes (B)
    hist2_luma: [u32; 256],
    hist2_r:    [u32; 256],
    hist2_g:    [u32; 256],
    hist2_b:    [u32; 256],

    // Chemins
    path_a: Option<PathBuf>,
    path_b: Option<PathBuf>,

    // Nom du dossier de rejet
    bin_folder_name: String,
    bin_folder_input : String,

    // Comparaison
    compare_enabled: bool,
    compare_mode: CompareMode,   
    compare_split: f32,
    compare_center_uv: [f32; 2],
    blink_hz: f32,               

    compare_spacing: f32, //espace entre les images (horz)
    compare_vertical_offset : f32, //espace entre les images (vert) avec cisaillement de A par rapport à B

    link_views: bool,

    show_split_divider: bool,

    // États de vue individuels (utilisés si link_views == false)
    zoom_a: f32,
    zoom_b: f32,
    offset_a: Vec2,  // en points, repère rect local
    offset_b: Vec2,

    pub max_tex_side_device: u32,

    //status message
    status_message: Option<(String, egui::Color32)>,  // texte + couleur
    status_timer: f32,     // temps restant (en secondes)
    
    //Metadata
    meta_loader: MetaLoader,
    meta_inflight_a: bool,
    meta_inflight_b: bool,
    meta_a: Vec<(String, String)>,
    meta_b: Vec<(String, String)>,
    meta_err_a: Option<String>,
    meta_err_b: Option<String>,

    // --- Modal "Image properties" ---
    show_props: bool,                 // ouverture/fermeture du popup
    props_tab: PropsTab,              // onglet actif
    props_a: Vec<(String, String)>,   // propriétés calculées pour A
    props_b: Vec<(String, String)>,   // idem B

    // Transition fade
    fade_alpha_a: f32,
    fade_alpha_b: f32,
    fade_start_a: Option<Instant>,
    fade_start_b: Option<Instant>,
    fade_enabled : bool,
    slideshow_fade_duration: f32,  
                      
}

impl Default for App {
    fn default() -> Self {
        let loader = start_loader(num_cpus::get().clamp(2, 8));
        Self {
            orig_a: None,
            size_a: [0, 0],
            tex_a_cpu: None,
            orig_b: None,
            size_b: [0, 0],
            tex_b_cpu: None,

            slideshow_mode: false,
            auto_slideshow: true,
            slideshow_interval: 3.0,
            slideshow_timer: 0.0,

            loader,
            next_req_id: 1,
            last_req_a: None,
            last_req_b: None,
            inflight_a : false,
            inflight_b : false,
            
            pending_free: Vec::new(),

            filelist_a: Vec::new(),
            idx_a: 0,
            filelist_b: Vec::new(),
            idx_b: 0,

            show_options: false,

            show_about: false,

            zoom: 1.0,
            min_zoom: 0.05,
            max_zoom: 100.0,
            offset: Vec2::ZERO,
            request_fit: false,
            request_center: false,
            request_one_to_one: false,
            keep_center_on_resize: true,
            auto_fit_on_transform: false,
            last_panel_size: None,

            subfolders: Vec::new(),
            show_new_folder_dialog: false,
            new_folder_input: String::new(),

            fit_allow_upscale: false,

            zoom_step_percent: 30.0,

            rotation : 0,
            flip_h: false,
            flip_v: false,

            linear_filter: true,

            bg_gray: 18,

            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            gamma: 1.0,

            hist_luma: [0; 256],
            hist_r: [0; 256],
            hist_g: [0; 256],
            hist_b: [0; 256],

            hist2_luma: [0; 256],
            hist2_r:    [0; 256],
            hist2_g:    [0; 256],
            hist2_b:    [0; 256],

            hist_dirty: false,
            log_hist: false,
            hist_rgb_mode: true,

            path_a: None,
            path_b: None,

            bin_folder_name: String::from("Visua_bin"),
            bin_folder_input: String::from("Visua_bin"),

            compare_enabled: false,
            compare_mode: CompareMode::Split,   // par défaut on reste en Split
            compare_split: 0.5,
            compare_center_uv: [0.5, 0.5],
            blink_hz: 2.0,                       // 2 Hz confortable

            compare_spacing: 0.0,
            compare_vertical_offset : 0.0,

            link_views: true,

            show_split_divider: true,

            zoom_a: 1.0,
            zoom_b: 1.0,
            offset_a: Vec2::ZERO,
            offset_b: Vec2::ZERO,

            max_tex_side_device: MAX_TEX_SIDE_FALLBACK,

            status_message: None,
            status_timer: 0.0,

            meta_loader: start_meta_loader(),
            meta_inflight_a: false,
            meta_inflight_b: false,
            meta_a: Vec::new(),
            meta_b: Vec::new(),
            meta_err_a: None,
            meta_err_b: None,

            show_props: false,
            props_tab: PropsTab::Properties,
            props_a: Vec::new(),
            props_b: Vec::new(),

            fade_alpha_a : 0.0,
            fade_alpha_b : 0.0,
            fade_start_a : None,
            fade_start_b : None,
            fade_enabled : true,
            slideshow_fade_duration : 0.65,        
        }
    }
}

impl App {

    fn refresh_subfolders(&mut self) {
        self.subfolders.clear();

        if let Some(img_path) = &self.path_a {
            if let Some(parent) = img_path.parent() {
                if let Ok(entries) = std::fs::read_dir(parent) {
                    for entry in entries.flatten() {
                        if entry.path().is_dir() {
                            if let Some(name) = entry.file_name().to_str() {
                                self.subfolders.push(name.to_string());
                            }
                        }
                    }
                }
            }
        }

        self.subfolders.sort();

        if !self.subfolders.contains(&self.bin_folder_name) {
            self.bin_folder_name.clear();
        }

    }

    fn set_status_message(&mut self, msg: &str, color: egui::Color32, duration: f32) {
        self.status_message = Some((msg.to_string(), color));
        self.status_timer = duration; // ex: 3.0 secondes
    }

    /// Déplace l'image courante (A) dans <dossier>/Visua_bin/ en générant un nom unique si besoin,
    /// puis charge automatiquement la suivante (sinon vide l'affichage).
    fn move_current_to_bin(&mut self, ctx: &egui::Context) -> Result<(), String> {

        if self.bin_folder_name.is_empty() {
            self.set_status_message(
                "⚠ Select or create a subfolder before sorting.",
                egui::Color32::RED,
                3.0,
            );
            return Err("Aucun dossier de tri sélectionné.".to_string());
        } else {

            let p = self.path_a.clone().ok_or("Aucune image chargée.")?;
            let dir = p.parent().ok_or("Chemin sans parent.")?;
            let bin = dir.join(self.bin_folder_name.as_str());
            fs::create_dir_all(&bin).map_err(|e| e.to_string())?;

            let name = p.file_name().ok_or("Nom de fichier invalide.")?;
            let mut dst = bin.join(name);

            // Nom unique si collision
            if dst.exists() {
                let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
                let ext  = p.extension().and_then(|s| s.to_str()).unwrap_or("");
                let mut i = 1;
                loop {
                    let cand = if ext.is_empty() {
                        bin.join(format!("{stem}_{i}"))
                    } else {
                        bin.join(format!("{stem}_{i}.{ext}"))
                    };
                    if !cand.exists() { dst = cand; break; }
                    i += 1;
                }
            }

            // Déplacement (rename) avec fallback copy+remove si autre volume
            match fs::rename(&p, &dst) {
                Ok(_) => {}
                Err(_) => {
                    fs::copy(&p, &dst).map_err(|e| e.to_string())?;
                    fs::remove_file(&p).map_err(|e| e.to_string())?;
                }
            }

            // Recharger la suivante dans le dossier  

            self.after_move_reload_next(ctx, dir, &p)?;
            
            Ok(())
        }
    }

    /// Après déplacement, ouvre la "suivante" du dossier (par nom) ; sinon nettoie l'affichage.
    fn after_move_reload_next(
        &mut self,
        ctx: &egui::Context,
        dir: &Path,
        old: &Path,
    ) -> Result<(), String> {
        // 1) Re-scan du dossier (sans l’ancien)
        let mut files: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| is_supported_ext(p))
            .collect();
        files.sort(); // tri alphabétique

        let old_name = old.file_name().unwrap_or_default();

       // 2) Choisir la “prochaine” (première strictement > old), sinon WRAP sur la première
        let mut next: Option<PathBuf> = None;
        for f in &files {
            if let Some(n) = f.file_name() {
                if n > old_name {
                    next = Some(f.clone());
                    break;
                }
            }
        }
        // wrap-around : si on supprimait la dernière -> revenir à la première
        if next.is_none() && !files.is_empty() {
            next = Some(files[0].clone());
        }

        // 3) Nettoyage des ressources A (on va potentiellement changer d’image)
        self.orig_a = None;
        if let Some(old) = self.tex_a_cpu.take() { self.pending_free.push(old); }

        // 4) Charger la nouvelle “courante” via le WRAPPER qui réaligne filelist_a + idx_a
        if let Some(n) = next {
            // ⬅️ CHANGEMENT MAJEUR: utiliser load_image_a (et non load_image_a_only)
            self.load_image_a(ctx, n)?;
        } else {
            // plus d’image dans le dossier → état propre
            self.path_a = None;
            self.size_a = [0, 0];
            self.hist_dirty = true;
            self.filelist_a.clear();
            self.idx_a = 0;
        }

        Ok(())
    }

    fn build_filelist_for(path: &Path) -> Vec<PathBuf> {
        let dir = path.parent().unwrap_or(Path::new("."));
        let mut files: Vec<PathBuf> = fs::read_dir(dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|s| s.to_str())
                    .map(supported_ext)
                    .unwrap_or(false)
            })
            .collect();

        files.sort_by(|a, b| {
            let sa = a
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase();
            let sb = b
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase();
            sa.cmp(&sb)
        });
        files
    }

    fn set_filelist_a(&mut self, p: &Path) {
        self.filelist_a = Self::build_filelist_for(p);
        self.idx_a = self.filelist_a.iter().position(|x| x == p).unwrap_or(0);
    }

    fn set_filelist_b(&mut self, p: &Path) {
        self.filelist_b = Self::build_filelist_for(p);
        self.idx_b = self.filelist_b.iter().position(|x| x == p).unwrap_or(0);
    }

    fn decide_load_size(&self, w: u32, h: u32) -> Result<(u32, u32, Option<f64>), String> {
            if w == 0 || h == 0 {
                return Err("dimensions nulles".into());
            }
            let max_side = self.max_tex_side_device.max(MAX_TEX_SIDE_FALLBACK);

            if w <= max_side && h <= max_side {
                return Ok((w, h, None));
            }

            // facteur minimal pour que max(w, h) ≤ max_side
            let f = (max_side as f64 / w as f64)
                .min(max_side as f64 / h as f64)
                .min(1.0);

            let tw = ((w as f64 * f).floor() as u32).max(1);
            let th = ((h as f64 * f).floor() as u32).max(1);

            Ok((tw, th, Some(f)))
        }

    /// Garde-fou juste avant la création de la texture GPU.
    fn validate_gpu_budget(&self, w: u32, h: u32) -> Result<(), String> {
        let max_side = self.max_tex_side_device.max(MAX_TEX_SIDE_FALLBACK);
        if w > max_side || h > max_side {
            return Err(format!("{}×{} dépasse la limite GPU {} px", w, h, max_side));
        }
        Ok(())
    }

    fn load_image_to_texture(
        ctx: &egui::Context,
        rgba_arc: &std::sync::Arc<Vec<u8>>,
        size: [usize; 2],
        name: &str,
        linear: bool,
    ) -> egui::TextureHandle {
        let [w, h] = size;
        // Construction d’un ColorImage (egui s’occupe de l’upload + padding 256B)
        let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba_arc[..]);
        let opts = if linear {
            egui::TextureOptions::LINEAR
        } else {
            egui::TextureOptions::NEAREST
        };
        ctx.load_texture(name.to_owned(), img, opts)
    }

    fn load_image_a_only(&mut self, ctx: &egui::Context, p: PathBuf) -> Result<(), String> {

        // 1) dimensions & décision taille (TIFF pris en charge)
        let ext = p.extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();

        // --- Cas général ---
        let (w0, h0) = if matches!(ext.as_str(), "tif" | "tiff") {
            peek_tiff_dims(&p).map_err(|e| e.to_string())?
        } else {
            image::image_dimensions(&p).map_err(|e| e.to_string())?
        };

        let (tw, th, maybe_f) = self.decide_load_size(w0, h0)?;

        // 2) décodage générique en RGBA8 (TIFF/Autres)
        let (mut data, dims) = load_any_rgba8(&p)?;
        let (w, h) = (dims[0] as u32, dims[1] as u32);

        // Éventuel resize Lanczos3 (après décodage)
        let rgba8: image::RgbaImage = if maybe_f.is_some() && (w != tw || h != th) {
            let img = image::RgbaImage::from_raw(w, h, std::mem::take(&mut data))
                .ok_or_else(|| "buffer RGBA invalide".to_string())?;
            let resized = image::DynamicImage::ImageRgba8(img)
                .resize_exact(tw, th, image::imageops::FilterType::Lanczos3);
            resized.to_rgba8()
        } else {
            image::RgbaImage::from_raw(w, h, data)
                .ok_or_else(|| "buffer RGBA invalide".to_string())?
        };
        let (w, h) = rgba8.dimensions();

        // 3) garde-fou texture côté GPU
        self.validate_gpu_budget(w, h)?;

        // 4) stockage CPU
        self.size_a = [w as usize, h as usize];
        self.orig_a = Some(std::sync::Arc::new(rgba8.into_raw()));
        self.path_a = Some(p);

        // 5) upload via egui (gère l'alignement 256B)
        let opts = if self.linear_filter { egui::TextureOptions::LINEAR } else { egui::TextureOptions::NEAREST };
        let data = self.orig_a.as_ref().unwrap();
        let tex = ctx.load_texture(
            "imgA_cpu",
            egui::ColorImage::from_rgba_unmultiplied([self.size_a[0], self.size_a[1]], &data[..]),
            opts,
        );
        if let Some(old) = self.tex_a_cpu.replace(tex) { self.pending_free.push(old); }

        // 6) marquages UI
        self.hist_dirty = true;
        self.request_center = true;
        self.compare_center_uv = [0.5, 0.5];
        self.cmd_fit();

        Ok(())
    }


    fn load_image_b_only(&mut self, ctx: &egui::Context, p: PathBuf) -> Result<(), String> {
        // 1) dimensions & décision taille (TIFF pris en charge)
        let ext = p.extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();

        // ---- CAS IMAGE CLASSIQUE  ----
        let (w0, h0) = if matches!(ext.as_str(), "tif" | "tiff") {
            peek_tiff_dims(&p).map_err(|e| e.to_string())?
        } else {
            image::image_dimensions(&p).map_err(|e| e.to_string())?
        };
        
        let (tw, th, maybe_f) = self.decide_load_size(w0, h0)?;

        // 2) décodage générique en RGBA8 (TIFF/Autres)
        let (mut data, dims) = load_any_rgba8(&p)?;
        let (w, h) = (dims[0] as u32, dims[1] as u32);

        // Éventuel resize Lanczos3 (après décodage)
        let rgba8: image::RgbaImage = if maybe_f.is_some() && (w != tw || h != th) {
            let img = image::RgbaImage::from_raw(w, h, std::mem::take(&mut data))
                .ok_or_else(|| "buffer RGBA invalide".to_string())?;
            let resized = image::DynamicImage::ImageRgba8(img)
                .resize_exact(tw, th, image::imageops::FilterType::Lanczos3);
            resized.to_rgba8()
        } else {
            image::RgbaImage::from_raw(w, h, data)
                .ok_or_else(|| "buffer RGBA invalide".to_string())?
        };
        let (w, h) = rgba8.dimensions();
        
        // 3) garde-fou texture côté GPU
        self.validate_gpu_budget(w, h)?;

        // 4) stockage CPU
        self.size_b = [w as usize, h as usize];
        self.orig_b = Some(std::sync::Arc::new(rgba8.into_raw()));
        self.path_b = Some(p);

        // 5) upload via egui
        let opts = if self.linear_filter { egui::TextureOptions::LINEAR } else { egui::TextureOptions::NEAREST };
        let data = self.orig_b.as_ref().unwrap();
        let tex = ctx.load_texture(
            "imgB_cpu",
            egui::ColorImage::from_rgba_unmultiplied([self.size_b[0], self.size_b[1]], &data[..]),
            opts,
        );
        if let Some(old) = self.tex_b_cpu.replace(tex) { self.pending_free.push(old); }

        // 6) marquages UI
        self.hist_dirty = true;
        self.request_center = true;
        self.compare_center_uv = [0.5, 0.5];
        self.cmd_fit();

        Ok(())
    }


    fn load_image_a(&mut self, _ctx: &egui::Context, p: PathBuf) -> Result<(), String> {
        // 🔹 Mets à jour la liste de fichiers et l’index courant
        self.set_filelist_a(&p);
        self.idx_a = self.filelist_a.iter().position(|x| x == &p).unwrap_or(0);
        self.path_a = Some(p.clone());

        // 🔹 Forcer un fit au premier affichage de l’image
        self.request_fit = true;

        // 🔹 Réinitialiser le fade à chaque nouvelle image
        self.fade_start_a = Some(Instant::now());
        self.fade_alpha_a = 0.0;

        // 🔹 Envoie du job asynchrone de chargement
        let max_side = self.max_tex_side_device;
        self.request_image_a(p, max_side);

        Ok(())
    }

    fn load_image_b(&mut self, _ctx: &egui::Context, p: PathBuf) -> Result<(), String> {
        self.set_filelist_b(&p);
        self.idx_b = self.filelist_b.iter().position(|x| x == &p).unwrap_or(0);
        self.path_b = Some(p.clone());
        self.request_fit = true;

        self.fade_start_b = Some(Instant::now());
        self.fade_alpha_b = 0.0;
       
        let max_side = self.max_tex_side_device;
        self.request_image_b(p, max_side);
        
        Ok(())
        
    }

    fn navigate_a(&mut self, ctx: &egui::Context, step: i32) -> Result<(), String> {
        if self.filelist_a.is_empty() {
            return Ok(());
        }
        let len = self.filelist_a.len() as i32;
        let mut ni = self.idx_a as i32 + step;
        if ni < 0 {
            ni = (ni % len + len) % len;
        } else {
            ni %= len;
        }
        self.idx_a = ni as usize;
        let path = self.filelist_a[self.idx_a].clone();
        self.path_a = Some(path.clone());

        self.fade_start_a = Some(Instant::now());
        self.fade_alpha_a = 0.0;

        self.load_image_a_only(ctx, path)

    }

    fn navigate_b(&mut self, ctx: &egui::Context, step: i32) -> Result<(), String> {
        if self.filelist_b.is_empty() {
            return Ok(());
        }
        let len = self.filelist_b.len() as i32;
        let mut ni = self.idx_b as i32 + step;
        if ni < 0 {
            ni = (ni % len + len) % len;
        } else {
            ni %= len;
        }
        self.idx_b = ni as usize;
        let path = self.filelist_b[self.idx_b].clone();
        self.path_b = Some(path.clone());

        self.fade_start_b = Some(Instant::now());
        self.fade_alpha_b = 0.0;

        self.load_image_b_only(ctx, path)
    }

    fn cmd_fit(&mut self) {
        self.request_fit = true;        
        self.request_center = true;
    }

    fn cmd_center(&mut self) {
        self.request_center = true;
    }

    fn cmd_one_to_one(&mut self) {
        self.request_fit = false;
        self.zoom = 1.0;                    
        self.request_center = true; 
        self.request_one_to_one = true;        
    }

    fn cmd_flip_h(&mut self) {
        self.flip_h = !self.flip_h;
    }

    fn cmd_flip_v(&mut self) {
        self.flip_v = !self.flip_v;
    }

    fn center_in(&mut self, panel_rect: Rect, tex_size: [usize; 2]) {
        if tex_size[0] == 0 || tex_size[1] == 0 {
            return;
        }
        let (w, h) = (tex_size[0] as f32 * self.zoom, tex_size[1] as f32 * self.zoom);
        let tl =
            panel_rect.left_top() + 0.5 * Vec2::new(panel_rect.width() - w, panel_rect.height() - h);
        self.offset = tl - panel_rect.left_top();
    }

    fn set_linear_filter(&mut self, ctx: &egui::Context, linear: bool) {
        if self.linear_filter != linear {
            self.linear_filter = linear;
            if let (Some(orig), Some(_)) = (&self.orig_a, &self.tex_a_cpu) {
                let h = Self::load_image_to_texture(ctx, &orig, self.size_a, "imgA_cpu", linear);
                if let Some(old) = self.tex_a_cpu.replace(h) { self.pending_free.push(old); }
            }
            if let (Some(orig), Some(_)) = (&self.orig_b, &self.tex_b_cpu) {
                let h = Self::load_image_to_texture(ctx, &orig, self.size_b, "imgB_cpu", linear);
                if let Some(old) = self.tex_b_cpu.replace(h) { self.pending_free.push(old); }
            }
        }
    }
    fn mark_hist_dirty(&mut self) {
        self.hist_dirty = true;
    }

    //Fonction pour fermer image b
    fn close_b(&mut self, ctx: &egui::Context) {
        // 1) Vider l’image B (CPU + GPU)
        self.orig_b = None;
        self.size_b = [0, 0];

        if let Some(tex) = self.tex_b_cpu.take() {
            self.pending_free.push(tex);
        }

        // 2) Reset état de navigation/selection B
        self.path_b = None;
        self.filelist_b = Vec::new();
        self.idx_b = 0;
        self.last_req_b = None;
        self.inflight_b = false;

        // 3) Désactiver comportements spécifiques à la comparaison + centrer
        self.compare_enabled = false;
        self.cmd_fit();         

        // 4) UI : une frame tout de suite
        ctx.request_repaint();
    }

    //Helper UI pour drainer les résultats 
    fn drain_loader(&mut self, ctx: &egui::Context) {
        let mut changed = false;

        while let Ok(msg) = self.loader.rx.try_recv() {
            match msg {
                JobResult::Ok { id, target, size, rgba } => {
                    let opts = if self.linear_filter { egui::TextureOptions::LINEAR }
                            else { egui::TextureOptions::NEAREST };

                    match target {
                        Target::MainPaneA => {
                            if Some(id) == self.last_req_a {
                                self.size_a = size;
                                self.orig_a = Some(rgba.clone());
                                let img = egui::ColorImage::from_rgba_unmultiplied(
                                    [size[0], size[1]], &rgba[..]
                                );
                                let tex = ctx.load_texture("imgA_cpu", img, opts);
                                if let Some(old) = self.tex_a_cpu.replace(tex) {
                                    self.pending_free.push(old);
                                }

                                // 🔹 Reset fade-in pour A
                                self.fade_start_a = Some(Instant::now());
                                self.fade_alpha_a = 0.0;

                                self.hist_dirty = true;
                                self.request_center = true;
                                self.compare_center_uv = [0.5, 0.5];

                                self.inflight_a = false;
                                changed = true;
                            }
                        }
                        Target::MainPaneB => {
                            if Some(id) == self.last_req_b {
                                self.size_b = size;
                                self.orig_b = Some(rgba.clone());
                                let img = egui::ColorImage::from_rgba_unmultiplied(
                                    [size[0], size[1]], &rgba[..]
                                );
                                let tex = ctx.load_texture("imgB_cpu", img, opts);
                                if let Some(old) = self.tex_b_cpu.replace(tex) {
                                    self.pending_free.push(old);
                                }

                                // 🔹 Reset fade-in pour B
                                self.fade_start_b = Some(Instant::now());
                                self.fade_alpha_b = 0.0;

                                self.hist_dirty = true;
                                self.request_center = true;
                                self.compare_center_uv = [0.5, 0.5];

                                self.inflight_b = false;
                                changed = true;
                            }
                        }
                    }
                }
                JobResult::Err { target, error, .. } => {
                    changed = true;
                    eprintln!("load error for {:?}: {}", target, error);
                }
            }
        }

        if changed {
            ctx.request_repaint(); // réveille l’UI immédiatement
        }
    }

    fn request_image_a(&mut self, path: PathBuf, max_side: u32) {
        let id = ReqId(self.next_req_id); self.next_req_id += 1;
        self.last_req_a = Some(id);
        self.inflight_a = true;
        self.orig_a = None;
        if let Some(old) = self.tex_a_cpu.take() {
            self.pending_free.push(old);
        }
        let max_side = max_side.min(MAX_SIDE);  // MAX_SIDE = 20_000 
        let _ = self.loader.tx.send(Job::Load {
            id, path, target: Target::MainPaneA, max_side
        });
    }

    fn request_image_b(&mut self, path: PathBuf, max_side: u32) {
        let id = ReqId(self.next_req_id); self.next_req_id += 1;
        self.last_req_b = Some(id);
        self.inflight_b = true;
        self.orig_b = None;
        if let Some(old) = self.tex_b_cpu.take() {
            self.pending_free.push(old);
        }
        let max_side = max_side.min(MAX_SIDE);  // MAX_SIDE = 20_000 
        let _ = self.loader.tx.send(Job::Load {
            id, path, target: Target::MainPaneB, max_side
        });
    }

}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {

        let now = Instant::now();

        let fade_duration = if self.slideshow_mode {
            self.slideshow_fade_duration.max(0.01)
        } else if self.fade_enabled {
            0.30// fade rapide hors slideshow
        } else {
            0.0 // pas de fade
        };

        if fade_duration > 0.0 {
            if let Some(start) = self.fade_start_a {
                let elapsed = now.duration_since(start).as_secs_f32();
                if elapsed < fade_duration {
                    self.fade_alpha_a = (elapsed / fade_duration).clamp(0.0, 1.0);
                    ctx.request_repaint();
                } else {
                    self.fade_alpha_a = 1.0;
                    self.fade_start_a = None;
                }
            }

            if let Some(start) = self.fade_start_b {
                let elapsed = now.duration_since(start).as_secs_f32();
                if elapsed < fade_duration {
                    self.fade_alpha_b = (elapsed / fade_duration).clamp(0.0, 1.0);
                    ctx.request_repaint();
                } else {
                    self.fade_alpha_b = 1.0;
                    self.fade_start_b = None;
                }
            }
        } else {
            // Pas de fade → alpha toujours plein
            self.fade_alpha_a = 1.0;
            self.fade_alpha_b = 1.0;
            self.fade_start_a = None;
            self.fade_start_b = None;
        }

        // resultats Exif
        while let Ok(msg) = self.meta_loader.rx.try_recv() {
            match msg {
                MetaResult::Ok { pane: Pane::A, items } => {
                    self.meta_a = items;
                    self.meta_err_a = None;
                    self.meta_inflight_a = false;
                }
                MetaResult::Ok { pane: Pane::B, items } => {
                    self.meta_b = items;
                    self.meta_err_b = None;
                    self.meta_inflight_b = false;
                }
                MetaResult::Err { pane: Pane::A, error } => {
                    self.meta_a.clear();
                    self.meta_err_a = Some(error);
                    self.meta_inflight_a = false;
                }
                MetaResult::Err { pane: Pane::B, error } => {
                    self.meta_b.clear();
                    self.meta_err_b = Some(error);
                    self.meta_inflight_b = false;
                }
            }
            ctx.request_repaint();
        }

        // Drainer les résultats des workers
        self.drain_loader(ctx);
        if self.inflight_a || self.inflight_b {
            ctx.request_repaint();
        }

        // sortie du diaporama avec Échap ou clic droit
        if self.slideshow_mode && (ctx.input(|i| i.key_pressed(egui::Key::Escape) || i.pointer.secondary_clicked())) {
            self.slideshow_mode = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
        }

        //Timer autoslideshow
        if self.slideshow_mode && self.auto_slideshow {
            let dt = ctx.input(|i| i.stable_dt); // delta temps frame
            self.slideshow_timer -= dt;

            if self.slideshow_timer <= 0.0 {
                let _ = self.navigate_a(ctx, 1); // avancer
                self.slideshow_timer = self.slideshow_interval; // reset
            }
            // 🔹 forcer un redraw même sans interaction
            ctx.request_repaint();
        }

        if self.slideshow_mode {
            // 1) Récupération des inputs → pas encore de mutation de self
            let nav_step = ctx.input(|i| {
                if i.key_pressed(egui::Key::ArrowRight)
                    || i.pointer.primary_clicked()     // clic gauche
                    || i.raw_scroll_delta.y < 0.0      // molette vers le haut
                {
                    Some(1)
                } else if i.key_pressed(egui::Key::ArrowLeft) || i.raw_scroll_delta.y > 0.0 {
                    Some(-1)
                } else {
                    None
                }
            });

            // 2) Navigation si input détecté
            if let Some(step) = nav_step {
                if let Err(e) = self.navigate_a(ctx, step) {
                    eprintln!("slideshow navigate error: {e}");
                } else {
                    // 🔹 reset du fade
                    self.fade_start_a = Some(Instant::now());
                    self.fade_alpha_a = 0.0;
                }

                if self.auto_slideshow {
                    self.slideshow_timer = self.slideshow_interval; // reset si navigation manuelle
                }
            }

            // 3) Affichage image A plein écran (avec GPU fade, fit et centrage)
            egui::CentralPanel::default().show(ctx, |ui| {
                let panel_rect = ui.max_rect();

                if let Some(pa) = &self.orig_a {
                    // dimensions de l’image source
                    let (tw, th) = (self.size_a[0] as f32, self.size_a[1] as f32);

                    // facteur "fit"
                    let fit = (panel_rect.width() / tw).min(panel_rect.height() / th);

                    // dimensions finales affichées
                    let draw_w = tw * fit;
                    let draw_h = th * fit;

                    // offset pour centrer
                    let offset_x = (panel_rect.width() - draw_w) * 0.5;
                    let offset_y = (panel_rect.height() - draw_h) * 0.5;

                    // paramètres GPU
                    let p = GpuParams {
                        brightness: self.brightness,
                        contrast: self.contrast,
                        saturation: self.saturation,
                        gamma: self.gamma,
                        flip_h: self.flip_h as u32,
                        flip_v: self.flip_v as u32,
                        rotation: self.rotation,
                        _pad0: 0,
                        zoom: fit,
                        _pad1: 0.0,
                        tex_w: tw,
                        tex_h: th,
                        off_x: offset_x,
                        off_y: offset_y,
                        center_u: -1.0,
                        center_v: -1.0,
                        rect_min_x: panel_rect.min.x,
                        rect_min_y: panel_rect.min.y,
                        rect_max_x: panel_rect.max.x,
                        rect_max_y: panel_rect.max.y,
                        ppp: 1.0,
                        fade_alpha: self.fade_alpha_a, // 🔹 fade dynamique
                        _pad2: [0.0; 3],
                        _pad3: [0.0; 4],
                    };

                    let cb = make_postprocess_paint_callback(
                        panel_rect,
                        Arc::clone(pa),
                        self.size_a,
                        self.linear_filter,
                        p,
                    );
                    ui.painter().add(cb);
                }
            });

            return; // bloque le reste de l'UI pendant le diaporama
        }

        // Drop différé des anciennes textures (celles collectées au frame précédent)
        self.pending_free.clear();

        // Détecte la vraie limite GPU si possible, sinon garde 8192
        if self.max_tex_side_device == MAX_TEX_SIDE_FALLBACK {
        }

        // >>> Force le rafraîchissement périodique en mode Blink
        if self.compare_enabled
            && matches!(self.compare_mode, CompareMode::Blink)
            && self.orig_a.is_some()
            && self.orig_b.is_some()
        {
            // demi-période = temps entre deux basculements A/B
            let half_period = (0.5 / self.blink_hz.max(0.01)) as f32; // en secondes
            ctx.request_repaint_after(Duration::from_secs_f32(half_period));
            // (alternative "toujours fluide" mais plus coûteuse : ctx.request_repaint();)
        }

        if self.show_about {
            let screen_rect = ctx.screen_rect();

            // 1) Zone invisible absorbant les clics (sous le popup)
            egui::Area::new(egui::Id::new("modal_blocker"))
                .order(egui::Order::Middle) // sous la fenêtre
                .fixed_pos(screen_rect.min)
                .show(ctx, |ui| {
                    ui.allocate_response(screen_rect.size(), egui::Sense::click());
                });

            // 2) Voile semi-transparent
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Background,
                egui::Id::new("about_modal_bg"),
            ));
            painter.rect_filled(screen_rect, 0.0, egui::Color32::from_black_alpha(100));

            // popup
            egui::Area::new(egui::Id::new("about_modal"))
                .order(egui::Order::Foreground) // reste toujours au-dessus
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    egui::Frame::window(&ctx.style()).show(ui, |ui| {
                        ui.heading("🖼 Visua – Image Viewer");
                        ui.separator();
                        ui.label(env!("CARGO_PKG_VERSION"));
                        ui.label("Author: AdrienLor");
                        ui.separator();
                        ui.label("Formats :");
                        ui.label("- PNG, JPG, BMP, WEBP, TGA, GIF, HDR, TIFF (incl. 32-bit float)");
                        ui.separator();
                        ui.label("Help – Slideshow controls:");
                        ui.label("• F11 : Enter slideshow fullscreen");
                        ui.label("• Esc or Right Click : Exit slideshow");
                        ui.label("• Right Arrow / Left Click / Scroll Down : Next image");
                        ui.label("• Left Arrow / Scroll Up : Previous image");
                        ui.separator();
                        if ui.button("Close").clicked() {
                            self.show_about = false;
                        }
                    });
                });            
        }

        // Barre du haut — actions
        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.set_min_height(26.0);
                    if ui.button("Open A").clicked()
                        || ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::O))
                    {
                        if let Some(path) = FileDialog::new()
                            .add_filter(
                                "Images",
                                &[
                                    "png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp", "gif", "tga",
                                    "ico", "pnm", "hdr"
                                ],
                            )
                            .pick_file()
                        {
                            let _ = self.load_image_a(ctx, path);
                        }
                    }
                    
                if self.orig_a.is_some(){
                    if ui.button("Open B").clicked() {
                        if let Some(path) = FileDialog::new()
                            .add_filter(
                                "Images",
                                &[
                                    "png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp", "gif", "tga",
                                    "ico", "pnm", "hdr"
                                ],
                            )
                            .pick_file()
                        {
                            let _ = self.load_image_b(ctx, path);
                            self.compare_enabled = true;
                            self.request_fit = true;
                            self.compare_center_uv = [0.5, 0.5];
                        }
                    }
                }

                //bouton close B
                if self.compare_enabled{
                    if ui.add(
                    egui::Button::new(
                        RichText::new("X").color(egui::Color32::from_rgb(255, 255, 255)))
                            .fill(egui::Color32::from_rgb(231, 52, 21))
                        ).clicked() {
                        self.close_b(ctx);
                    }
                }

                ui.separator();
                let disable_a = self.filelist_a.len() <= 1;
                let disable_b = self.filelist_b.len() <= 1;

                
                // Navigation dossier A
                if ui
                    .add_enabled(!disable_a, egui::Button::new("A ◀ Prev."))
                    .clicked()
                {
                    let _ = self.navigate_a(ctx, -1);
                }
                if ui
                    .add_enabled(!disable_a, egui::Button::new("A Next ▶"))
                    .clicked()
                {
                    let _ = self.navigate_a(ctx, 1);
                }
            

                // ---- Navigation pour B (si compare_enabled ) ----
                if self.compare_enabled {
                    // Navigation dossier B (comme avant)
                    if ui
                        .add_enabled(!disable_b, egui::Button::new("B ◀ Prev."))
                        .clicked()
                    {
                        let _ = self.navigate_b(ctx, -1);
                    }
                    if ui
                        .add_enabled(!disable_b, egui::Button::new("B Next ▶"))
                        .clicked()
                    {
                        let _ = self.navigate_b(ctx, 1);
                    }
                }
                

                //Bouton Diaporama et options connexes
                if !self.compare_enabled && self.orig_a.is_some() {
                    ui.separator();
                    if ui.button("🎞 Slideshow").clicked() {
                        self.slideshow_mode = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(true));
                    }

                    if ui.checkbox(&mut self.auto_slideshow, "Auto").changed() {
                        if self.auto_slideshow {
                            self.slideshow_timer = self.slideshow_interval;
                        }
                    }

                    ui.add(
                    egui::Slider::new(&mut self.slideshow_interval, 1.0..=30.0)
                        .text("sec.")
                    );
                    
                    ui.add(
                        egui::Slider::new(&mut self.slideshow_fade_duration, 0.0..=5.0)
                            .text("Fade")
                    );
                }
        
                // toggle diaporama avec F11
                if ctx.input(|i| i.key_pressed(egui::Key::F11)) {
                    self.slideshow_mode = !self.slideshow_mode;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.slideshow_mode));
                }

                ui.separator();

                // --- mémorise l'état avant interaction ---
                let was_compare_enabled = self.compare_enabled;
                let was_mode = self.compare_mode;
                let was_link = self.link_views;
                
                if self.compare_enabled && self.orig_a.is_some() && self.orig_b.is_some() {

                    
                ui.horizontal(|ui| {
                        // Combo compact sans label pour gagner de la place :
                        egui::ComboBox::from_id_salt("cmp_mode")
                            .selected_text(self.compare_mode.label())
                            .show_ui(ui, |ui| {
                                let style = ui.style_mut();
                                style.spacing.item_spacing.x = 4.0;
                                style.spacing.item_spacing.y = 4.0;
                                ui.selectable_value(&mut self.compare_mode, CompareMode::Split, "Side/Side");
                                ui.selectable_value(&mut self.compare_mode, CompareMode::Blink, "Blink");
                                ui.selectable_value(&mut self.compare_mode, CompareMode::Diff,  "Diff");
                                // ui.selectable_value(&mut self.compare_mode, CompareMode::Overlay, "Overlay");
                                // ui.selectable_value(&mut self.compare_mode, CompareMode::Swipe,   "Swipe");
                            });

                        // (Option) petit bouton aide/tooltip :
                        // ui.small_button("ⓘ").on_hover_text(self.compare_mode.help());
                    });
                
                    ui.separator();

                    if self.compare_mode==CompareMode::Split { 
                        ui.checkbox(&mut self.show_split_divider, "Div.");
                        ui.checkbox(&mut self.link_views, "Link");
                        ui.separator();
                    }

                    // si changement d'état Link → initialise proprement les paramètres
                    if self.link_views != was_link {
                        if self.link_views {
                            // On repasse en mode lié : unifie le zoom sur la base des deux
                            let avg = 0.5 * (self.zoom_a + self.zoom_b);
                            self.zoom = avg.clamp(self.min_zoom, self.max_zoom);
                            self.offset = Vec2::ZERO;
                            self.compare_center_uv = [0.5, 0.5];
                            self.request_center = true;
                        } else {
                            // On passe en mode indépendant : duplique les valeurs actuelles
                            self.zoom_a = self.zoom.clamp(self.min_zoom, self.max_zoom);
                            self.zoom_b = self.zoom_a;
                            self.offset_a = Vec2::ZERO;
                            self.offset_b = Vec2::ZERO;
                            self.request_center = true;
                        }
                    }

                    match self.compare_mode {
                        CompareMode::Split => {
                            if self.link_views {
                                //ui.add(egui::Slider::new(&mut self.compare_split, 0.0..=1.0).text("Split (A ⇠ B)"));
                                ui.add(egui::Slider::new(&mut self.compare_spacing, -1.0..=1.0).step_by(0.001).drag_value_speed(0.001).text("Horz."));
                                ui.add(egui::Slider::new(&mut self.compare_vertical_offset, -1.0..=1.0).step_by(0.001).drag_value_speed(0.001).text("Vert."));
                                if ui.button("Reset").clicked() {
                                    self.compare_spacing = 0.000;
                                    self.compare_vertical_offset = 0.000;
                                } 
                            }            
                        }
                        CompareMode::Blink => {
                            ui.add(egui::Slider::new(&mut self.blink_hz, 0.5..=8.0).text("Hz").logarithmic(true));
                            // --- PATCH: si on vient d'activer Blink, centre l'image ---
                            let blink_just_enabled =
                                self.compare_enabled
                                && (!was_compare_enabled || was_mode != self.compare_mode)
                                && matches!(self.compare_mode, CompareMode::Blink);

                            if blink_just_enabled {
                                // centre au prochain frame (utilise ta logique existante dans le CentralPanel)
                                self.compare_center_uv = [0.5, 0.5];
                                self.request_center = true;
                                ui.ctx().request_repaint();
                            }
                        }
                        CompareMode::Diff => {
                            let diff_just_enabled =
                                self.compare_enabled
                                && (!was_compare_enabled || was_mode != self.compare_mode)
                                && matches!(self.compare_mode, CompareMode::Diff);

                            if diff_just_enabled {
                                // centre au prochain frame (utilise ta logique existante dans le CentralPanel)
                                self.compare_center_uv = [0.5, 0.5];
                                self.request_center = true;
                                ui.ctx().request_repaint();
                            }
                        }
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // -- Bouton Options
                        if ui.add(
                    egui::Button::new(
                        RichText::new("⚙").color(egui::Color32::from_rgb(255, 255, 255)))
                            .fill(egui::Color32::from_rgb(243, 172, 17))
                        ).clicked() {
                            self.show_options = !self.show_options;
                        }
                    // -- Bouton About
                        if ui.add(
                    egui::Button::new(
                        RichText::new("ℹ").color(egui::Color32::from_rgb(255, 255, 255)))
                            .fill(egui::Color32::from_rgb(231, 52, 21))
                        ).clicked() {
                            self.show_props = true;

                            if let Some(p) = &self.path_a {
                                self.props_a = build_properties_for(self.size_a, p);
                                let _ = self.meta_loader.tx.send(MetaJob { pane: Pane::A, path: p.clone() });
                                self.meta_inflight_a = true;
                            }
                            if self.compare_enabled {
                                if let Some(p) = &self.path_b {
                                    self.props_b = build_properties_for(self.size_b, p);
                                    let _ = self.meta_loader.tx.send(MetaJob { pane: Pane::B, path: p.clone() });
                                    self.meta_inflight_b = true;
                                }
                            }
                        }

                    // Boutons FIT 11 center     
                    ui.separator();
                    ui.horizontal(|ui| {

                        if ui.button("[+]").clicked() {
                            self.cmd_center();
                        }
                        if ui.button("1:1").clicked() {
                            self.cmd_one_to_one();
                        }
                        if ui.button("Fit").clicked() {
                            self.fit_allow_upscale = true;
                            self.cmd_fit();
                        }
                    });
                });
                });
            });          
        });

        // Seconde ligne — chemins et bouton option
        // egui::TopBottomPanel::top("file_info_line").min_height(23.0).show(ctx, |ui| {
        //     ui.horizontal_wrapped(|ui| {
        //         ui.small(match (&self.path_a, &self.path_b) {
        //             (Some(pa), Some(pb)) => format!("A: {}    |    B: {}", pa.display(), pb.display()),
        //             (Some(pa), None) => format!("A: {}", pa.display()),
        //             (None, Some(pb)) => format!("B: {}", pb.display()),
        //             _ => String::from("—"),
        //         });
        //     });
        // });

        // Panneau outils (droite)
        if self.show_options {
            egui::SidePanel::right("tools").default_width(268.0).resizable(false).show(ctx, |ui| {
                ui.add_space(4.0);
                ui.heading("View");
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("⟳90°").clicked() {
                        self.rotation = (self.rotation + 90) % 360;
                        self.request_center = true;
                    }
                    if ui.button("↔ Horz.").clicked() {
                        self.cmd_flip_h();
                    }
                    if ui.button("↕ Vert.").clicked() {
                        self.cmd_flip_v();
                    }
                    if ui.button("Reset").clicked() {
                        self.rotation = 0;
                        self.flip_h = false;
                        self.flip_v = false;
                        self.request_center = true;
                    }
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Smoothing");
                    let mut linear = self.linear_filter;
                    if ui.checkbox(&mut linear, "").changed() {
                        self.set_linear_filter(ctx, linear);
                    }
                    ui.label("Fade");
                    ui.checkbox(&mut self.fade_enabled, "");
                });

                ui.add_space(8.0);
                let mut bg_i = self.bg_gray as i32;
                if ui.add(egui::Slider::new(&mut bg_i, 0..=255).step_by(1.0).drag_value_speed(1.0).text("Background color")).changed() 
                {
                    self.bg_gray = bg_i as u8;
                }

                ui.add_space(8.0);
                ui.separator();
                ui.heading("Adjustments");
                ui.add_space(8.0);
                let (mut b, mut c, mut s, mut g) =
                    (self.brightness, self.contrast, self.saturation, self.gamma);
                if ui
                    .add(egui::Slider::new(&mut b, -1.0..=1.0).text("Brightness").step_by(0.001).drag_value_speed(0.001))
                    .changed()
                {
                    self.brightness = b;
                    self.mark_hist_dirty();
                }
                if ui
                    .add(egui::Slider::new(&mut c, 0.0..=2.0).text("Contrast").step_by(0.001).drag_value_speed(0.001))
                    .changed()
                {
                    self.contrast = c;
                    self.mark_hist_dirty();
                }
                if ui
                    .add(egui::Slider::new(&mut s, 0.0..=2.0).text("Saturation").step_by(0.001).drag_value_speed(0.001))
                    .changed()
                {
                    self.saturation = s;
                    self.mark_hist_dirty();
                }
                
                if ui
                    .add(egui::Slider::new(&mut g, 0.2..=3.0).text("Gamma").step_by(0.001).drag_value_speed(0.001))
                    .changed()
                {
                    self.gamma = g;
                    self.mark_hist_dirty();
                }
                ui.add_space(8.0);
                if ui.button("Reset").clicked() {
                    self.brightness = 0.0;
                    self.contrast = 1.0;
                    self.saturation = 1.0;
                    self.gamma = 1.0;
                    self.mark_hist_dirty();
                }
                ui.add_space(8.0);

                ui.separator();
                if self.orig_a.is_some() {
                    ui.heading("Histogram A");
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.hist_rgb_mode, false, "Luma");
                        ui.radio_value(&mut self.hist_rgb_mode, true, "RGB");
                    });
                    ui.checkbox(&mut self.log_hist, "Log");
                    ui.add_space(6.0);
                    if self.hist_rgb_mode {
                        draw_histogram_rgb(ui, &self.hist_r, &self.hist_g, &self.hist_b, 120.0, self.log_hist);
                    } else {
                        draw_histogram_luma(ui, &self.hist_luma, 120.0, self.log_hist);
                    }
                }
                // Histogramme (B) si comparaison active et image B chargée
                if self.compare_enabled && self.orig_b.is_some() {
                    ui.add_space(6.0);
                    ui.heading("Histogram B");
                    if self.hist_rgb_mode {
                        draw_histogram_rgb(ui, &self.hist2_r, &self.hist2_g, &self.hist2_b, 120.0, self.log_hist);
                    } else {
                        draw_histogram_luma(ui, &self.hist2_luma, 120.0, self.log_hist);
                    }
                }

                if !self.compare_enabled && self.orig_a.is_some() {
                    ui.separator(); 
                    ui.heading("Sorting");
                    ui.add_space(6.0);

                    // Toujours rafraîchir la liste
                    self.refresh_subfolders();

                    egui::ComboBox::from_label("Subfolders")
                        .selected_text(
                            if self.bin_folder_name.is_empty() {
                                "Select…"
                            } else {
                                &self.bin_folder_name
                            }
                        )
                        .show_ui(ui, |ui| {
                            if self.subfolders.is_empty() {
                                // Aucun sous-dossier → forcer la création
                                ui.label("⚠ No subfolders available");
                            } else {
                                // Sous-dossiers existants
                                for folder in &self.subfolders {
                                    if ui.selectable_label(self.bin_folder_name == *folder, folder).clicked() {
                                        self.bin_folder_name = folder.clone();
                                    }
                                }
                            }

                            ui.separator();
                            if ui.button("➕ Create new subfolder…").clicked() {
                                self.show_new_folder_dialog = true;
                                self.new_folder_input.clear();
                            }
                        });

                    ui.add_space(12.0);
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Sort").color(egui::Color32::LIGHT_GRAY)
                            )
                            .fill(egui::Color32::from_rgb(231, 52, 21))
                        )                        
                        .clicked()
                    {
                        if let Err(e) = self.move_current_to_bin(ctx) {
                            eprintln!("[bin] {}", e); 
                        }
                    }

                }
                ui.with_layout(egui::Layout::bottom_up(egui::Align::RIGHT), |ui| {
                    ui.add_space(8.0);
                    if ui.button("About").clicked() {
                        self.show_about = true;
                    }
                    
                });
              

            });
        }

        // Popup pour créer un nouveau dossier
        if self.show_new_folder_dialog {
            let screen_rect = ctx.screen_rect();

            // 1) zone invisible absorbant les clics
            egui::Area::new(egui::Id::new("new_folder_blocker"))
                .order(egui::Order::Middle)
                .fixed_pos(screen_rect.min)
                .show(ctx, |ui| {
                    ui.allocate_response(screen_rect.size(), egui::Sense::click());
                });

            // 2) voile semi-transparent
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Background,
                egui::Id::new("new_folder_modal_bg"),
            ));
            painter.rect_filled(screen_rect, 0.0, egui::Color32::from_black_alpha(100));

            // 3) fenêtre modale (toujours au-dessus)
            egui::Area::new(egui::Id::new("new_folder_modal"))
                .order(egui::Order::Foreground)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    egui::Frame::window(&ctx.style()).show(ui, |ui| {
                        ui.heading("New Subfolder");
                        ui.separator();

                        ui.label("Subfolder name (A–Z, a–z, 0–9, _ , -) :");
                        let resp = ui.text_edit_singleline(&mut self.new_folder_input);

                        if resp.changed() {
                            self.new_folder_input.retain(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
                            if self.new_folder_input.len() > 40 {
                                self.new_folder_input.truncate(40);
                            }
                        }

                        ui.horizontal(|ui| {
                            if ui.button("Create").clicked() && !self.new_folder_input.is_empty() {
                                if let Some(img_path) = &self.path_a {
                                    if let Some(parent) = img_path.parent() {
                                        let new_dir = parent.join(&self.new_folder_input);
                                        if let Err(e) = std::fs::create_dir_all(&new_dir) {
                                            eprintln!("Erreur création dossier: {e}");
                                        } else {
                                            self.bin_folder_name = self.new_folder_input.clone();
                                            self.refresh_subfolders();
                                        }
                                    }
                                }
                                self.show_new_folder_dialog = false;
                            }

                            if ui.button("Cancel").clicked() {
                                self.show_new_folder_dialog = false;
                            }
                        });
                    });
                });

            // 4) fermeture via Esc ou clic droit
            if ctx.input(|i| i.key_pressed(egui::Key::Escape) || i.pointer.secondary_clicked()) {
                self.show_new_folder_dialog = false;
            }
        }

        if self.show_props {
            // --- voile sombre (bloque l'arrière-plan) ---
            let screen = ctx.input(|i| i.screen_rect());
            let blocker = egui::Rect::from_min_size(screen.left_top(), screen.size());
            egui::Area::new(egui::Id::new("props_modal_blocker"))
                .order(egui::Order::Background)
                .interactable(true)
                .fixed_pos(screen.left_top())
                .show(ctx, |ui| {
                    let _ = ui.interact(blocker, ui.id().with("catch_all"), egui::Sense::click());
                    ui.painter()
                        .rect_filled(blocker, 0.0, egui::Color32::from_black_alpha(100));
                });

            // --- fenêtre centrale ---
            egui::Window::new("Image properties")
                // ⬇️ pas de .open()
                .collapsible(false)
                .resizable(false)
                .default_size([640.0, 480.0])
                .constrain(true)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        let _ = ui.selectable_value(&mut self.props_tab, PropsTab::Properties, "Properties");
                        let _ = ui.selectable_value(&mut self.props_tab, PropsTab::Exif, "EXIF");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Close").clicked() {
                                self.show_props = false; // ferme manuellement
                            }
                        });
                    });
                    ui.separator();

                    let dual = self.compare_enabled && self.path_b.is_some();
                    match self.props_tab {
                        PropsTab::Properties => {
                            ui_props_table(ui, &self.props_a, &self.props_b, dual);
                        }
                        PropsTab::Exif => {
                            ui_props_exif(
                                ui,
                                &self.meta_a,
                                &self.meta_b,
                                dual,
                                self.meta_inflight_a,
                                self.meta_inflight_b,
                                self.meta_err_a.as_deref(),
                                self.meta_err_b.as_deref(),
                            );
                        }
                    };
                });
        }

        // Panneau central
        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let (panel_rect, response) =
                ui.allocate_exact_size(avail, Sense::drag().union(Sense::hover()));
            if panel_rect.width() <= 1.0 || panel_rect.height() <= 1.0 {
                return; // fenêtre trop petite / redimensionnement -> pas de dessin
            }

            // split pour comparaison
            // split pour comparaison (évite les extrêmes 0.0 / 1.0)
            let mut s = self.compare_split.clamp(0.0, 1.0);
            let eps = 0.001;
            if s <= eps { s = eps; }
            if s >= 1.0 - eps { s = 1.0 - eps; }

            let split_x = panel_rect.left() + panel_rect.width() * s;
            let left_rect  = egui::Rect::from_min_max(panel_rect.left_top(), egui::pos2(split_x, panel_rect.bottom()));
            let right_rect = egui::Rect::from_min_max(egui::pos2(split_x, panel_rect.top()), panel_rect.right_bottom());

            // fit/center
            if self.compare_enabled {
                match self.compare_mode {
                    CompareMode::Split => {
                        if !self.link_views {
                            // ---------- Link OFF : contrôles indépendants A / B ----------

                            // FIT par pane
                            if self.request_fit {
                                // A
                                if self.size_a != [0, 0] {
                                    // 🔑 inverser largeur/hauteur si rotation est 90 ou 270
                                    let (w, h) = if self.rotation % 180 == 0 {
                                        (self.size_a[0] as f32, self.size_a[1] as f32)
                                    } else {
                                        (self.size_a[1] as f32, self.size_a[0] as f32)
                                    };

                                    let fit_a = (left_rect.width() / w).min(left_rect.height() / h);
                                    let target_a = if self.fit_allow_upscale { fit_a } else { fit_a.min(1.0) };
                                    self.zoom_a = target_a.max(0.05);

                                    let draw_a = egui::Vec2::new(w * self.zoom_a, h * self.zoom_a);
                                    self.offset_a = 0.5 * (left_rect.size() - draw_a);
                                } else {
                                    self.zoom_a = 1.0;
                                    self.offset_a = egui::Vec2::ZERO;
                                }
                                                                // B
                                if self.size_b != [0, 0] {
                                    // 🔑 inverser largeur/hauteur si rotation_b est 90 ou 270
                                    let (w, h) = if self.rotation % 180 == 0 {
                                        (self.size_b[0] as f32, self.size_b[1] as f32)
                                    } else {
                                        (self.size_b[1] as f32, self.size_b[0] as f32)
                                    };

                                    let fit_b = (right_rect.width() / w).min(right_rect.height() / h);
                                    let target_b = if self.fit_allow_upscale { fit_b } else { fit_b.min(1.0) };
                                    self.zoom_b = target_b.max(0.05);

                                    let draw_b = egui::Vec2::new(w * self.zoom_b, h * self.zoom_b);
                                    self.offset_b = 0.5 * (right_rect.size() - draw_b);
                                } else {
                                    self.zoom_b = 1.0;
                                    self.offset_b = egui::Vec2::ZERO;
                                }

                                // min_zoom cohérent avec les deux
                                self.min_zoom = (self.zoom_a.min(self.zoom_b) * 0.001).max(0.005);
                                self.request_center = true;
                                self.request_fit = false;
                                ctx.request_repaint();
                            }

                           // --- 1:1 indépendant (Link OFF) ---
                            if self.request_one_to_one {
                                self.zoom_a = 1.0;
                                self.zoom_b = 1.0;

                                // centre visuel dans chaque sous-rect
                                let draw_a = egui::Vec2::new(self.size_a[0] as f32 * self.zoom_a, self.size_a[1] as f32 * self.zoom_a);
                                let draw_b = egui::Vec2::new(self.size_b[0] as f32 * self.zoom_b, self.size_b[1] as f32 * self.zoom_b);

                                self.offset_a = 0.5 * (left_rect.size()  - draw_a);
                                self.offset_b = 0.5 * (right_rect.size() - draw_b);

                                // sécurité si une image manque
                                if self.size_a == [0,0] { self.offset_a = egui::Vec2::ZERO; }
                                if self.size_b == [0,0] { self.offset_b = egui::Vec2::ZERO; }

                                // ne pas enclencher de fit derrière :
                                self.request_fit = false;
                                self.fit_allow_upscale = false;
                                self.request_one_to_one = false;

                                ctx.request_repaint();
                            }

                            // Center indépendant
                            if self.request_center {
                                if self.size_a != [0, 0] {
                                    let (w, h) = if self.rotation % 180 == 0 {
                                        (self.size_a[0] as f32, self.size_a[1] as f32)
                                    } else {
                                        (self.size_a[1] as f32, self.size_a[0] as f32)
                                    };
                                    let draw_a = egui::Vec2::new(w * self.zoom_a, h * self.zoom_a);
                                    self.offset_a = 0.5 * (left_rect.size() - draw_a);
                                } else {
                                    self.offset_a = egui::Vec2::ZERO;
                                }

                                if self.size_b != [0, 0] {
                                    let (w, h) = if self.rotation % 180 == 0 {
                                        (self.size_b[0] as f32, self.size_b[1] as f32)
                                    } else {
                                        (self.size_b[1] as f32, self.size_b[0] as f32)
                                    };
                                    let draw_b = egui::Vec2::new(w * self.zoom_b, h * self.zoom_b);
                                    self.offset_b = 0.5 * (right_rect.size() - draw_b);
                                } else {
                                    self.offset_b = egui::Vec2::ZERO;
                                }

                                self.request_center = false;
                                ctx.request_repaint();
                            }

                        } else {
                            // ---------- Link ON : ton comportement actuel (partagé) ----------
                            if self.request_fit {
                                let mut scales: Vec<f32> = vec![];

                                if self.size_a != [0, 0] {
                                    let (w, h) = if self.rotation % 180 == 0 {
                                        (self.size_a[0] as f32, self.size_a[1] as f32)
                                    } else {
                                        (self.size_a[1] as f32, self.size_a[0] as f32)
                                    };
                                    scales.push((left_rect.width() / w).min(left_rect.height() / h));
                                }

                                if self.size_b != [0, 0] {
                                    let (w, h) = if self.rotation % 180 == 0 {
                                        (self.size_b[0] as f32, self.size_b[1] as f32)
                                    } else {
                                        (self.size_b[1] as f32, self.size_b[0] as f32)
                                    };
                                    scales.push((right_rect.width() / w).min(right_rect.height() / h));
                                }

                                if let Some(mins) = scales.into_iter().reduce(f32::min) {
                                    let target = if self.fit_allow_upscale { mins } else { mins.min(1.0) };
                                    self.zoom = target.max(0.05);
                                    self.min_zoom = (self.zoom * 0.001).max(0.005);

                                    // 🔑 RECALCULER LE CENTRAGE ICI
                                    let (w, h) = if self.rotation % 180 == 0 {
                                        (self.size_a[0] as f32, self.size_a[1] as f32)
                                    } else {
                                        (self.size_a[1] as f32, self.size_a[0] as f32)
                                    };
                                    let tw = w as f32 * self.zoom;
                                    let th = h as f32 * self.zoom;
                                    let offset_x = (panel_rect.width() - tw) * 0.5;
                                    let offset_y = (panel_rect.height() - th) * 0.5;
                                    self.offset = egui::vec2(offset_x, offset_y);
                                }

                                self.request_fit = false;
                            }

                            if self.request_one_to_one {
                                self.zoom = 1.0;
                                self.compare_center_uv = [0.5, 0.5];
                                self.request_fit = false;
                                self.fit_allow_upscale = false;
                                self.request_one_to_one = false;
                            }

                            if self.request_center {
                                self.compare_center_uv = [0.5, 0.5];
                                self.request_center = false;
                            }
                        }
                    }
                    // Blink et Diff: fit comme en mode normal, sur tout le panel
                    _ => {
                        let ref_tex = if self.size_a != [0, 0] { self.size_a } else { self.size_b };
                        if self.request_fit && ref_tex != [0, 0] {
                            // 🔑 inverser w/h si rotation en 90 ou 270
                            let (w, h) = if self.rotation % 180 == 0 {
                                (ref_tex[0] as f32, ref_tex[1] as f32)
                            } else {
                                (ref_tex[1] as f32, ref_tex[0] as f32)
                            };

                            let zx = panel_rect.width() / w;
                            let zy = panel_rect.height() / h;
                            let fit = zx.min(zy);

                            let target = if self.fit_allow_upscale { fit } else { fit.min(1.0) };

                            self.zoom = target.max(0.05);
                            self.min_zoom = (self.zoom * 0.001).max(0.005);

                            // ⚠️ self.center_in doit aussi recevoir les dimensions corrigées
                            self.center_in(panel_rect, [w as usize, h as usize]);

                            self.request_fit = false;
                        }
                        // 1:1 intelligent
                        if self.request_one_to_one {
                            self.zoom = 1.0;
                            // recentrer (à adapter à tes champs : center UV partagé ou par-slot)
                            self.compare_center_uv = [0.5, 0.5];
                            // self.offset = egui::vec2(0.0, 0.0);  // si tu as un pan en pixels

                            self.request_fit = false;
                            self.fit_allow_upscale = false;

                            self.request_one_to_one = false;
                        }

                        if self.request_center && ref_tex != [0, 0] {
                            let (w, h) = if self.rotation % 180 == 0 {
                                (ref_tex[0], ref_tex[1])
                            } else {
                                (ref_tex[1], ref_tex[0])
                            };
                            self.center_in(panel_rect, [w, h]);
                            self.request_center = false;
                        }
                    }
                }
                } else {
                let size_now = panel_rect.size();
                if self.keep_center_on_resize {
                    if let Some(prev) = self.last_panel_size {
                        if (prev.x - size_now.x).abs() > f32::EPSILON
                            || (prev.y - size_now.y).abs() > f32::EPSILON
                        {
                            self.request_center = true;
                        }
                    }
                }
                self.last_panel_size = Some(size_now);
                let ref_tex = if self.size_a != [0, 0] {
                    self.size_a
                } else {
                    self.size_b
                };

                if self.request_fit && ref_tex != [0, 0] {
                     let (w, h) = if self.rotation % 180 == 0 {
                        (ref_tex[0] as f32, ref_tex[1] as f32)
                    } else {
                        (ref_tex[1] as f32, ref_tex[0] as f32)
                    };

                    let zx = panel_rect.width() / w;
                    let zy = panel_rect.height() / h;
                    let fit = zx.min(zy);

                    let target = if self.fit_allow_upscale { fit } else { fit.min(1.0) };
                    self.zoom = target.max(0.05);
                    self.min_zoom = (self.zoom * 0.001).max(0.005);

                    // ✅ centrage UV plutôt que offset
                    self.request_center = true;
                    self.request_fit = false;
                }
                
                // 1:1 intelligent en mode simple
                if self.request_one_to_one && ref_tex != [0, 0] {
                     // VRAI 1:1 : 1 pixel image = 1 pixel écran, recentré, sans upscale forcé
                    self.zoom = 1.0;
                    // recentrage 
                    self.request_center = true;       // si tu utilises un recentrage différé
                   
                    self.request_fit = false;
                    self.fit_allow_upscale = false;

                    self.request_one_to_one = false;
                }

                if self.request_center && ref_tex != [0, 0] {
                    let (w, h) = if self.rotation % 180 == 0 {
                        (ref_tex[0], ref_tex[1])
                    } else {
                        (ref_tex[1], ref_tex[0])
                    };
                    self.center_in(panel_rect, [w, h]);
                    self.request_center = false;
                }
            }

            // ====== interactions =======

             // Raccourcis navigation
            let left = ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft));
            let right = ctx.input(|i| i.key_pressed(egui::Key::ArrowRight));
            let up = ctx.input(|i| i.key_pressed(egui::Key::ArrowUp));
            let down = ctx.input(|i| i.key_pressed(egui::Key::ArrowDown));
            
            if !self.show_about && !self.show_props {
                if left  {let _ = self.navigate_a(ctx, -1);}      
                if right {let _ = self.navigate_a(ctx, 1);}
                if up {let _ = self.navigate_b(ctx, -1);}
                if down {let _ = self.navigate_b(ctx, 1);}
            }

            match (self.compare_enabled, self.compare_mode) {
                (true, CompareMode::Split) => {
                    // Pan: on déplace le centre UV
                    if response.dragged() {
                        let rsz = if self.size_a != [0, 0] { self.size_a } else { self.size_b };
                        if rsz != [0, 0] {
                            let dd = response.drag_delta();
                            let du = -dd.x / (self.zoom * rsz[0] as f32);
                            let dv = -dd.y / (self.zoom * rsz[1] as f32);
                            self.compare_center_uv[0] = (self.compare_center_uv[0] + du).clamp(0.0, 1.0);
                            self.compare_center_uv[1] = (self.compare_center_uv[1] + dv).clamp(0.0, 1.0);
                        }
                    }
                    // Zoom: roue
                    if response.hovered() {
                        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                        if scroll != 0.0 {
                            let lines = (scroll / 120.0_f32).clamp(-10.0, 10.0);
                            let base: f32 = 1.0 + self.zoom_step_percent / 100.0;
                            self.zoom = (self.zoom * base.powf(lines)).clamp(self.min_zoom, self.max_zoom);
                        }
                    }
                }
                (true, _) => {
                    // Blink / Diff → pan/zoom comme en mode simple
                    if response.dragged() {
                        self.offset += response.drag_delta();
                    }
                    if response.hovered() {
                        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                        if scroll != 0.0 {
                            let lines = (scroll / 120.0_f32).clamp(-10.0, 10.0);
                            let base: f32 = 1.0 + self.zoom_step_percent / 100.0;
                            let old = self.zoom;
                            let new = (self.zoom * base.powf(lines)).clamp(self.min_zoom, self.max_zoom);
                            if (new - old).abs() > f32::EPSILON {
                                if let Some(mouse) = ui.input(|i| i.pointer.hover_pos()) {
                                    let tl = panel_rect.left_top() + self.offset;
                                    let delta_before = mouse - tl;
                                    let scale = new / old;
                                    let new_tl = mouse - delta_before * scale;
                                    self.offset += new_tl - tl;
                                }
                                self.zoom = new;
                            }
                        }
                    }
                }
                (false, _) => {
                    // 🔧 MODE NORMAL (comparaison off) — pan + zoom ancré sous la souris
                    let del_pressed = ctx.input(|i| i.key_pressed(egui::Key::Delete));
                    if del_pressed && !self.compare_enabled && !ui.ctx().wants_keyboard_input() {
                        if let Err(e) = self.move_current_to_bin(ctx) {
                            eprintln!("[bin] {}", e);
                        }
                    }
                                
                    if response.dragged() {
                        self.offset += response.drag_delta();
                    }

                    if response.hovered() {
                        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                        if scroll != 0.0 {
                            let lines = (scroll / 120.0_f32).clamp(-10.0, 10.0);
                            let base: f32 = 1.0 + self.zoom_step_percent / 100.0;
                            let old = self.zoom;
                            let new = (self.zoom * base.powf(lines)).clamp(self.min_zoom, self.max_zoom);
                            if (new - old).abs() > f32::EPSILON {
                                if let Some(mouse) = ui.input(|i| i.pointer.hover_pos()) {
                                    let tl = panel_rect.left_top() + self.offset;
                                    let delta_before = mouse - tl;
                                    let scale = new / old;
                                    let new_tl = mouse - delta_before * scale;
                                    self.offset += new_tl - tl;
                                }
                                self.zoom = new;
                            }
                        }
                    }
                }
            }

            // fond
            let painter = ui.painter_at(panel_rect);
            painter.rect_filled(panel_rect, 0.0, egui::Color32::from_gray(self.bg_gray));

            // Params GPU
            let build_params_simple = |tex_size: [usize; 2], fade: f32| -> GpuParams {
                GpuParams {
                    brightness: self.brightness,
                    contrast: self.contrast,
                    saturation: self.saturation,
                    gamma: self.gamma,
                    flip_h: self.flip_h as u32,
                    flip_v: self.flip_v as u32,
                    rotation : self.rotation,
                    _pad0: 0,
                    zoom: self.zoom,
                    _pad1: 0.0,
                    tex_w: tex_size[0] as f32,
                    tex_h: tex_size[1] as f32,
                    off_x: self.offset.x, // EN POINTS
                    off_y: self.offset.y, // EN POINTS
                    center_u: -1.0,
                    center_v: -1.0,
                    rect_min_x: 0.0,
                    rect_min_y: 0.0,
                    rect_max_x: 0.0,
                    rect_max_y: 0.0,
                    ppp: 1.0,
                    fade_alpha: fade,
                    _pad2: [0.0; 3],
                    _pad3: [0.0; 4],  
                }
            };

            // Draw via WGPU callbacks
            if self.compare_enabled && self.orig_a.is_some() && self.orig_b.is_some() {
                match self.compare_mode {
                    CompareMode::Split => {

                        // A à gauche
                        if let Some(pa) = &self.orig_a {
                            let mut p = build_params_simple(self.size_a, self.fade_alpha_a);

                            if self.link_views {
                                // 🔗 Mode LIÉ → zoom partagé
                                let (w, h) = if self.rotation % 180 == 0 {
                                    (self.size_a[0] as f32, self.size_a[1] as f32)
                                } else {
                                    (self.size_a[1] as f32, self.size_a[0] as f32)
                                };
                                let draw = egui::vec2(w * self.zoom, h * self.zoom);
                                let offset = 0.5 * (left_rect.size() - draw);

                                p.zoom = self.zoom;
                                p.off_x = offset.x - self.compare_spacing * left_rect.width() * 0.5;
                                p.off_y = offset.y - self.compare_vertical_offset * left_rect.height() * 0.5;
                            } else {
                                // 🔓 Mode indépendant
                                p.zoom = self.zoom_a;
                                p.off_x = self.offset_a.x - self.compare_spacing * left_rect.width() * 0.5;
                                p.off_y = self.offset_a.y - self.compare_vertical_offset * left_rect.height() * 0.5;
                            }

                            let cb = make_postprocess_paint_callback(left_rect, Arc::clone(pa), self.size_a, self.linear_filter, p);
                            ui.painter().add(cb);
                        }

                        // B à droite
                        if let Some(pb) = &self.orig_b {
                            let mut p = build_params_simple(self.size_b, self.fade_alpha_b);

                            if self.link_views {
                                // 🔗 Mode LIÉ → zoom partagé
                                let (w, h) = if self.rotation % 180 == 0 {
                                    (self.size_b[0] as f32, self.size_b[1] as f32)
                                } else {
                                    (self.size_b[1] as f32, self.size_b[0] as f32)
                                };
                                let draw = egui::vec2(w * self.zoom, h * self.zoom);
                                let offset = 0.5 * (right_rect.size() - draw);

                                p.zoom = self.zoom;
                                p.off_x = offset.x + self.compare_spacing * right_rect.width() * 0.5;
                                p.off_y = offset.y + self.compare_vertical_offset * right_rect.height() * 0.5;
                            } else {
                                // 🔓 Mode indépendant
                                p.zoom = self.zoom_b;
                                p.off_x = self.offset_b.x + self.compare_spacing * right_rect.width() * 0.5;
                                p.off_y = self.offset_b.y + self.compare_vertical_offset * right_rect.height() * 0.5;
                            }

                            let cb = make_postprocess_paint_callback(right_rect, Arc::clone(pb), self.size_b, self.linear_filter, p);
                            ui.painter().add(cb);
                        }

                        if self.show_split_divider {
                            let split_x = left_rect.right(); // frontière A|B
                            let divider = egui::Rect::from_min_max(
                                egui::pos2(split_x - 0.5, panel_rect.top()),
                                egui::pos2(split_x + 0.5, panel_rect.bottom()),
                            );
                            // rendu de la ligne
                            ui.painter().rect_filled(divider, 0.0, egui::Color32::from_white_alpha(180));
                        }


                        let mouse_pos = ui.input(|i| i.pointer.hover_pos());
                        let hover_a = mouse_pos.map(|p| left_rect.contains(p)).unwrap_or(false);
                        let hover_b = mouse_pos.map(|p| right_rect.contains(p)).unwrap_or(false);

                        // Pan
                        if response.dragged() {
                            let d = response.drag_delta();
                            if hover_a { self.offset_a += d; }
                            if hover_b { self.offset_b += d; }
                            ctx.request_repaint();
                        }

                        // Zoom (molette ancrée sous le curseur)
                        if response.hovered() {
                            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                            if scroll != 0.0 {
                                let lines = (scroll / 120.0_f32).clamp(-10.0, 10.0);
                                let base: f32 = 1.0 + self.zoom_step_percent / 100.0;

                                if hover_a {
                                    let old = self.zoom_a;
                                    let new = (self.zoom_a * base.powf(lines)).clamp(self.min_zoom, self.max_zoom);
                                    if (new - old).abs() > f32::EPSILON {
                                        if let Some(mp) = mouse_pos {
                                            let tl = left_rect.left_top() + self.offset_a;
                                            let before = mp - tl;
                                            let scale = new / old;
                                            let new_tl = mp - before * scale;
                                            self.offset_a += new_tl - tl;
                                        }
                                        self.zoom_a = new;
                                        ctx.request_repaint();
                                    }
                                } else if hover_b {
                                    let old = self.zoom_b;
                                    let new = (self.zoom_b * base.powf(lines)).clamp(self.min_zoom, self.max_zoom);
                                    if (new - old).abs() > f32::EPSILON {
                                        if let Some(mp) = mouse_pos {
                                            let tl = right_rect.left_top() + self.offset_b;
                                            let before = mp - tl;
                                            let scale = new / old;
                                            let new_tl = mp - before * scale;
                                            self.offset_b += new_tl - tl;
                                        }
                                        self.zoom_b = new;
                                        ctx.request_repaint();
                                    }
                                }
                            }
                        }

                    }
                    CompareMode::Blink => {
                        let t = ui.input(|i| i.time);
                        let phase = ((t * self.blink_hz as f64) as i64) & 1 == 0;
                        if phase {
                            // montre A plein cadre
                            if let Some(pa) = &self.orig_a {
                                let mut p = build_params_simple(self.size_a, self.fade_alpha_a);
                                p.rect_min_x = panel_rect.min.x; p.rect_min_y = panel_rect.min.y;
                                p.rect_max_x = panel_rect.max.x; p.rect_max_y = panel_rect.max.y;
                                let cb = make_postprocess_paint_callback(panel_rect, Arc::clone(pa), self.size_a, self.linear_filter, p);
                                ui.painter().add(cb);
                            }
                        } else {
                            if let Some(pb) = &self.orig_b {
                                let mut p = build_params_simple(self.size_b, self.fade_alpha_b);
                                p.rect_min_x = panel_rect.min.x; p.rect_min_y = panel_rect.min.y;
                                p.rect_max_x = panel_rect.max.x; p.rect_max_y = panel_rect.max.y;
                                let cb = make_postprocess_paint_callback(panel_rect, Arc::clone(pb), self.size_b, self.linear_filter, p);
                                ui.painter().add(cb);
                            }
                        }
                    }
                    CompareMode::Diff => {
                        if let (Some(pa), Some(pb)) = (&self.orig_a, &self.orig_b) {
                            let mut p = build_params_simple(self.size_a, self.fade_alpha_a); // on s'aligne sur A pour zoom/off_x/off_y
                            p.rect_min_x = panel_rect.min.x; p.rect_min_y = panel_rect.min.y;
                            p.rect_max_x = panel_rect.max.x; p.rect_max_y = panel_rect.max.y;
                            let cb = make_postprocess_paint_callback_diff(
                                panel_rect,
                                Arc::clone(pa), self.size_a,
                                Arc::clone(pb), self.size_b,
                                self.linear_filter,
                                p,
                            );
                            ui.painter().add(cb);
                        }
                    }
                }
            } else if let Some(pa) = &self.orig_a {
                let p = build_params_simple(self.size_a, self.fade_alpha_a);
                let cb = make_postprocess_paint_callback(panel_rect, Arc::clone(pa), self.size_a, self.linear_filter, p);
                ui.painter().add(cb);
            } else if let Some(pb) = &self.orig_b {
                let p = build_params_simple(self.size_b, self.fade_alpha_b);
                let cb = make_postprocess_paint_callback(panel_rect, Arc::clone(pb), self.size_b, self.linear_filter, p);
                ui.painter().add(cb);
            } else {
                // message vide
                painter.text(
                    panel_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "",
                    egui::TextStyle::Heading.resolve(ui.style()),
                    egui::Color32::from_gray(180),
                );
            }
            // barre d'état
            let status_rect = Rect::from_min_size(
                panel_rect.left_bottom() - Vec2::new(0.0, 24.0),
                Vec2::new(panel_rect.width(), 24.0),
            );
            ui.allocate_new_ui(UiBuilder::new().max_rect(status_rect), |ui| {
                ui.horizontal(|ui| {
                    if self.compare_enabled && !self.link_views {
                        ui.label(format!("A: {:.0}% | B: {:.0}%", self.zoom_a*100.0, self.zoom_b*100.0));
                    } else {
                        let z = (self.zoom * 100.0).round();
                        ui.label(format!("Zoom: {z:.0}%"));
                    }
                    if self.size_a[0] > 0 {
                        ui.separator();
                        ui.label(format!("A: {}×{}", self.size_a[0], self.size_a[1]));
                    }
                    if self.compare_enabled && self.size_b[0] > 0 {
                        ui.separator();
                        ui.label(format!("B: {}×{}", self.size_b[0], self.size_b[1]));
                    }
                     ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if let Some((msg, color)) = &self.status_message {
                            egui::Frame::none()
                                .fill(egui::Color32::WHITE)   // fond blanc
                                .inner_margin(egui::Margin::symmetric(8.0, 4.0))
                                .show(ui, |ui| {
                                    ui.colored_label(*color, msg);
                                });
                        } else {
                            ui.label(" "); // ligne vide si pas de message
                        }
                    });
                });
            });
        });

        //Timer pour le status message
        let dt = ctx.input(|i| i.stable_dt); // temps frame
        // décrémentation + auto-suppression
        if self.status_timer > 0.0 {
            self.status_timer -= dt;
            if self.status_timer <= 0.0 {
                self.status_message = None;
            } else {
                // 🔹 Tant qu’un message est affiché → on redessine
                ctx.request_repaint();
            }
        }

       // Histogrammes (A et B)
        if self.hist_dirty {
            let (l, r, g, b) = compute_hist_or_zero(
                self.orig_a.as_ref(),
                self.size_a,
                self.brightness, self.contrast, self.saturation, self.gamma,
                1_000_000,
            );
            self.hist_luma = l; self.hist_r = r; self.hist_g = g; self.hist_b = b;

            let (l2, r2, g2, b2) = compute_hist_or_zero(
                self.orig_b.as_ref(),
                self.size_b,
                self.brightness, self.contrast, self.saturation, self.gamma,
                1_000_000,
            );
            self.hist2_luma = l2; self.hist2_r = r2; self.hist2_g = g2; self.hist2_b = b2;

            self.hist_dirty = false;
        }

        // 🔹 Gestion du fade-in progressif pour A
        if self.fade_alpha_a < 1.0 {
            let dt = ctx.input(|i| i.stable_dt); // temps écoulé depuis la frame précédente
            self.fade_alpha_a += dt * 2.0; // vitesse du fade (ici ~0.5 sec)
            if self.fade_alpha_a > 1.0 {
                self.fade_alpha_a = 1.0;
            }
            ctx.request_repaint();
        }

        // 🔹 Gestion du fade-in progressif pour B
        if self.fade_alpha_b < 1.0 {
            let dt = ctx.input(|i| i.stable_dt);
            self.fade_alpha_b += dt * 2.0;
            if self.fade_alpha_b > 1.0 {
                self.fade_alpha_b = 1.0;
            }
            ctx.request_repaint();
        }

    }
}

fn main() -> eframe::Result<()> {
    use std::{env, ffi::OsString, path::PathBuf};

    let icon = {
        // chemin indépendant de l’emplacement du fichier source
        let bytes = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/app_256.png"));
        let image = image::load_from_memory(bytes).expect("icon png").to_rgba8();
        let (w, h) = image.dimensions();
        IconData { rgba: image.into_raw(), width: w, height: h }
    };

    let native_opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Visua")
            .with_inner_size(egui::vec2(1440.0, 820.0))
            .with_min_inner_size(egui::vec2(1440.0, 820.0))       
            .with_icon(icon),
        centered: true, // centrer à l’ouverture
        ..Default::default()
    };

    install_panic_hook_once();

    // Récupère les arguments passés par Windows (après l'exe)
    let args: Vec<OsString> = env::args_os().skip(1).collect();

    eframe::run_native(
        "Visua",
        native_opts,
        Box::new(move |cc| {
            // Démarre l'app normalement
            let mut app = App::default();

            // Si un fichier a été fourni => ouvre-le dans A
            if let Some(first) = args.get(0) {
                let p = PathBuf::from(first);
                if p.exists() {
                    // Appelle ton chargeur existant
                    let _ = app.load_image_a(&cc.egui_ctx, p.clone());
                }
            }

            Ok(Box::new(app))
        }),
    )
}