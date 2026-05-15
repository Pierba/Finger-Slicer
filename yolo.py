import cv2
import numpy as np
import os
from ultralytics import FastSAM

def extract_objects(image_path, output_dir):
    print("Caricamento del modello FastSAM...")
    # FastSAM scaricherà in automatico FastSAM-s.pt (leggero per CPU)
    model = FastSAM('FastSAM-s.pt') 

    img = cv2.imread(image_path)
    if img is None:
        print(f"Errore: Impossibile caricare l'immagine: {image_path}")
        return

    H, W, _ = img.shape
    os.makedirs(output_dir, exist_ok=True)

    print("Ricerca oggetti in corso...")
    # conf=0.4 evita che ritagli troppi artefatti o riflessi minuscoli
    results = model(image_path, conf=0.4, iou=0.9) 
    result = results[0]

    if result.masks is None:
        print("Nessun oggetto rilevato.")
        return

    boxes = result.boxes.xyxy.cpu().numpy()
    masks = result.masks.data.cpu().numpy()

    print(f"Trovati {len(masks)} oggetti prima del filtro di contenimento.")

    # --- Filtro di contenimento ---
    # Ridimensiona tutte le maschere alle dimensioni originali per il confronto
    masks_full = []
    for mask in masks:
        masks_full.append(cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST))
    masks_full = np.array(masks_full)

    # Calcola l'area di ciascuna maschera e dell'immagine
    areas = np.array([m.sum() for m in masks_full])
    image_area = H * W

    # --- Filtro area: scarta sfondo (troppo grande) e rumore (troppo piccolo) ---
    max_area_ratio = 0.60  # Maschere che coprono >60% dell'immagine = sfondo
    min_area_ratio = 0.005  # Maschere che coprono <0.5% dell'immagine = rumore
    discard = set()
    for i in range(len(areas)):
        ratio = areas[i] / image_area
        if ratio > max_area_ratio:
            print(f"  Maschera {i} scartata: sfondo ({ratio:.1%} dell'immagine)")
            discard.add(i)
        elif ratio < min_area_ratio:
            print(f"  Maschera {i} scartata: troppo piccola ({ratio:.2%} dell'immagine)")
            discard.add(i)

    # --- Filtro di contenimento: scarta sotto-maschere ---
    containment_threshold = 0.80  # Se l'80%+ dell'area è dentro un'altra maschera, scarta
    for i in range(len(masks_full)):
        if i in discard:
            continue
        for j in range(len(masks_full)):
            if i == j or j in discard:
                continue
            # Controlla se la maschera più piccola è contenuta nella più grande
            if areas[j] >= areas[i]:
                continue  # j è più grande o uguale, non è una sotto-maschera di i
            # j è più piccola di i: verifica se j è contenuta in i
            overlap = np.logical_and(masks_full[i] > 0.5, masks_full[j] > 0.5).sum()
            ratio = overlap / max(areas[j], 1)
            if ratio >= containment_threshold:
                discard.add(j)

    # Filtra maschere e bounding box
    keep_indices = [i for i in range(len(masks)) if i not in discard]
    masks = masks[keep_indices]
    boxes = boxes[keep_indices]
    print(f"Dopo il filtro: {len(masks)} oggetti rimasti (scartati {len(discard)}).")

    for i, (mask, box) in enumerate(zip(masks, boxes)):
        
        # Ridimensiona la maschera alle dimensioni originali dell'immagine
        mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Crea il canale Alpha (Trasparenza)
        alpha_channel = (mask_resized * 255).astype(np.uint8)

        # Unisci RGB + Alpha
        b, g, r = cv2.split(img)
        rgba_image = cv2.merge([b, g, r, alpha_channel])

        # Ritaglia usando il bounding box
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        cropped_object = rgba_image[y1:y2, x1:x2]

        # Salva senza nome classe, usando solo un contatore
        filename = os.path.join(output_dir, f"oggetto_{i}.png")
        cv2.imwrite(filename, cropped_object)
        print(f"Salvato: {filename}")

    print(f"Finito! Controlla la cartella '{output_dir}'.")