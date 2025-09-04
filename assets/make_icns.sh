#!/bin/bash
set -e

# Vérifie qu'un fichier PNG est donné
if [ -z "$1" ]; then
  echo "Usage: $0 <icon.png>"
  exit 1
fi

SRC="$1"
BASENAME=$(basename "$SRC" .png)
ICONSET="${BASENAME}.iconset"

# Crée le dossier temporaire
mkdir -p "$ICONSET"

# Génère toutes les tailles requises
sips -z 16 16     "$SRC" --out "$ICONSET/icon_16x16.png"
sips -z 32 32     "$SRC" --out "$ICONSET/icon_16x16@2x.png"
sips -z 32 32     "$SRC" --out "$ICONSET/icon_32x32.png"
sips -z 64 64     "$SRC" --out "$ICONSET/icon_32x32@2x.png"
sips -z 128 128   "$SRC" --out "$ICONSET/icon_128x128.png"
sips -z 256 256   "$SRC" --out "$ICONSET/icon_128x128@2x.png"
sips -z 256 256   "$SRC" --out "$ICONSET/icon_256x256.png"
sips -z 512 512   "$SRC" --out "$ICONSET/icon_256x256@2x.png"
sips -z 512 512   "$SRC" --out "$ICONSET/icon_512x512.png"
cp "$SRC" "$ICONSET/icon_512x512@2x.png"

# Génère le fichier .icns final
iconutil -c icns "$ICONSET" -o "${BASENAME}.icns"

# Nettoie le dossier temporaire
rm -rf "$ICONSET"

echo "✅ Icône générée : ${BASENAME}.icn
