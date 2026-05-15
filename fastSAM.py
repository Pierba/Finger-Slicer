import cv2
import numpy as np
import os
from ultralytics import FastSAM


# --- Parametri di filtraggio ---
MIN_AREA_RATIO = 0.005       # Maschere < 0.5% dell'immagine = rumore
MAX_AREA_RATIO = 0.60        # Maschere > 60% dell'immagine = sfondo
CONTAINMENT_THRESHOLD = 0.80 # Overlap >= 80% = sotto-maschera, scartata


def _filter_masks(masks_bin, boxes, image_area):
    """
    Filtra le maschere in 3 fasi:
      1. Scarta sfondo (area troppo grande)
      2. Scarta rumore (area troppo piccola)
      3. Scarta sotto-maschere contenute in maschere più grandi (containment)
    
    Args:
        masks_bin: array booleano (N, H, W) - maschere binarizzate
        boxes:     array float   (N, 4)    - bounding box [x1, y1, x2, y2]
        image_area: int - H * W dell'immagine originale

    Returns:
        keep: lista di indici da mantenere
    """
    n = len(masks_bin)
    
    # Area di ciascuna maschera (somma pixel True)
    areas = masks_bin.reshape(n, -1).sum(axis=1)
    ratios = areas / image_area

    # Fase 1-2: filtro per area (vettorizzato)
    valid = (ratios >= MIN_AREA_RATIO) & (ratios <= MAX_AREA_RATIO)
    
    n_bg = int((ratios > MAX_AREA_RATIO).sum())
    n_small = int((ratios < MIN_AREA_RATIO).sum())
    if n_bg:
        print(f"  Scartate {n_bg} maschere di sfondo (>{MAX_AREA_RATIO:.0%} immagine)")
    if n_small:
        print(f"  Scartate {n_small} maschere troppo piccole (<{MIN_AREA_RATIO:.1%} immagine)")

    # Indici candidati, ordinati per area decrescente (le più grandi prima)
    candidates = np.where(valid)[0]
    order = np.argsort(-areas[candidates])
    candidates = candidates[order]

    # Fase 3: filtro di contenimento
    # Pre-check veloce con bounding box prima del confronto pixel-level
    keep = []
    discarded = set()

    for idx in candidates:
        if idx in discarded:
            continue
        keep.append(idx)

        # Cerca sotto-maschere tra i candidati rimanenti (più piccole di idx)
        bx1, by1, bx2, by2 = boxes[idx]
        for other in candidates:
            if other in discarded or other == idx:
                continue
            if areas[other] >= areas[idx]:
                continue  # other non può essere sotto-maschera (è più grande)

            # Pre-filtro con bounding box: se il box di other non è contenuto
            # nel box di idx, non vale la pena fare il confronto pixel-level
            ox1, oy1, ox2, oy2 = boxes[other]
            # Calcola overlap dei box
            ix1 = max(bx1, ox1)
            iy1 = max(by1, oy1)
            ix2 = min(bx2, ox2)
            iy2 = min(by2, oy2)
            if ix1 >= ix2 or iy1 >= iy2:
                continue  # Nessun overlap nei box

            box_intersection = (ix2 - ix1) * (iy2 - iy1)
            box_other_area = (ox2 - ox1) * (oy2 - oy1)
            if box_other_area > 0 and box_intersection / box_other_area < CONTAINMENT_THRESHOLD:
                continue  # Overlap box insufficiente, skip confronto costoso

            # Confronto pixel-level (solo nella regione di intersezione dei box)
            iy1i, iy2i = int(iy1), int(iy2)
            ix1i, ix2i = int(ix1), int(ix2)
            overlap = np.count_nonzero(
                masks_bin[idx, iy1i:iy2i, ix1i:ix2i] & masks_bin[other, iy1i:iy2i, ix1i:ix2i]
            )
            if overlap / max(areas[other], 1) >= CONTAINMENT_THRESHOLD:
                discarded.add(other)

    print(f"  Scartate {len(discarded)} sotto-maschere (contenute in maschere più grandi)")
    return keep


def extract_objects(image_path, output_dir):
    print("Caricamento del modello FastSAM...")
    model = FastSAM('FastSAM-s.pt')

    img = cv2.imread(image_path)
    if img is None:
        print(f"Errore: Impossibile caricare l'immagine: {image_path}")
        return

    H, W, _ = img.shape
    os.makedirs(output_dir, exist_ok=True)

    print("Ricerca oggetti in corso...")
    results = model(image_path, conf=0.4, iou=0.9)
    result = results[0]

    if result.masks is None:
        print("Nessun oggetto rilevato.")
        return

    raw_masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    print(f"Trovate {len(raw_masks)} maschere grezze.")

    # Ridimensiona tutte le maschere a (H, W) e binarizza una sola volta
    masks_full = np.empty((len(raw_masks), H, W), dtype=bool)
    for i, m in enumerate(raw_masks):
        masks_full[i] = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5

    # Filtra maschere
    keep = _filter_masks(masks_full, boxes, H * W)
    print(f"Risultato: {len(keep)} oggetti validi.\n")

    # Prepara i canali BGR una sola volta (evita split ripetuto nel loop)
    b, g, r = cv2.split(img)

    # Estrai e salva gli oggetti
    for out_idx, mask_idx in enumerate(keep):
        alpha = (masks_full[mask_idx].astype(np.uint8)) * 255
        rgba = cv2.merge([b, g, r, alpha])

        # Ritaglia al bounding box con clamp ai bordi immagine
        x1, y1, x2, y2 = boxes[mask_idx].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        cropped = rgba[y1:y2, x1:x2]
        filename = os.path.join(output_dir, f"oggetto_{out_idx}.png")
        cv2.imwrite(filename, cropped)
        print(f"Salvato: {filename}")

    print(f"\nFinito! {len(keep)} oggetti nella cartella '{output_dir}'.")
