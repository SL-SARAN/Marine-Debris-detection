import cv2
import numpy as np
from pathlib import Path

base = Path('d:/MarineDebris/unprocessed_data/cylinder')

manta_items = []
cyl_items = []

for txt in base.glob('*/**/labels/*.txt'):
    img_p = txt.parent.parent / 'images' / (txt.stem + '.jpg')
    if not img_p.exists(): continue
    for l in txt.read_text().strip().splitlines():
        parts = [float(p) for p in l.split()]
        if len(parts) < 5: continue
        cls = int(parts[0])
        xc, yc, w, h = parts[1:5]
        bw = w * 1280
        bh = h * 1280
        elong = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 1.0
        ar = bw / bh if bh > 0 else 1.0
        item = {
            'file': txt.stem,
            'img_path': str(img_p),
            'cls': cls,
            'bw': bw, 'bh': bh,
            'elong': elong, 'ar': ar,
            'xc': xc, 'yc': yc, 'w': w, 'h': h
        }
        if cls == 1:
            manta_items.append(item)
        elif cls == 0:
            cyl_items.append(item)

print('Total Manta instances:', len(manta_items))
print('Total Cylider instances:', len(cyl_items))

manta_items.sort(key=lambda x: x['elong'], reverse=True)
print('\nTop 10 Most Elongated Manta:')
for m in manta_items[:10]:
    fn = m['file'][:32]
    print(f"  {fn}: elong={m['elong']:.2f}, ar={m['ar']:.2f}, {m['bw']:.1f}x{m['bh']:.1f}px")

# Check class confusion candidates
# Manta samples that have elongated rectangular or rod-like highlight vs cylinder samples that are truncated-cone
high_elong_manta = [m for m in manta_items if m['elong'] >= 1.8]
print(f"\nManta instances with elongation >= 1.8: {len(high_elong_manta)}")
for m in high_elong_manta:
    print(f"  {m['file'][:35]}: elong={m['elong']:.2f}, bw={m['bw']:.1f}, bh={m['bh']:.1f}")

# Cross-dataset overlap check
# Compare cylinder2 images against MILCO/NOMBO images in unprocessed_data (2010, 2015, 2017, 2018, 2021)
# and KLSG images in data/KLSG
print('\n' + '='*60)
print('CROSS-DATASET OVERLAP CHECK')
print('='*60)

import hashlib

def get_hash(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

cyl_hashes = {}
for p in base.glob('*/**/images/*.jpg'):
    cyl_hashes[get_hash(p)] = p

print(f'Hashed {len(cyl_hashes)} cylinder images.')

# Compare against MILCO (unprocessed_data/2010..2021)
milco_base = Path('d:/MarineDebris/unprocessed_data')
milco_matches = []
for y in ['2010', '2015', '2017', '2018', '2021']:
    ydir = milco_base / y
    if not ydir.exists(): continue
    for p in ydir.glob('*.jpg'):
        h = get_hash(p)
        if h in cyl_hashes:
            milco_matches.append((p, cyl_hashes[h]))

print(f'Exact SHA-256 matches between cylinder2 and MILCO (2010-2021): {len(milco_matches)}')

# Compare against KLSG
klsg_base = Path('d:/MarineDebris/data/KLSG')
klsg_matches = []
if klsg_base.exists():
    for p in klsg_base.glob('*/**/images/*.jpg'):
        h = get_hash(p)
        if h in cyl_hashes:
            klsg_matches.append((p, cyl_hashes[h]))
print(f'Exact SHA-256 matches between cylinder2 and KLSG: {len(klsg_matches)}')

# Check perceptual / downscaled similarity or filename patterns between datasets
print('\nFilename pattern check:')
cyl_names = [p.name for p in base.glob('*/**/images/*.jpg')]
print('Cylinder2 sample filenames:', cyl_names[:3])
for y in ['2010', '2015', '2017', '2018', '2021']:
    sample = list((milco_base / y).glob('*.jpg'))[:2]
    print(f'Year {y} sample filenames:', [s.name for s in sample])

if klsg_base.exists():
    sample_klsg = list(klsg_base.glob('*/**/images/*.jpg'))[:2]
    print('KLSG sample filenames:', [s.name for s in sample_klsg])
