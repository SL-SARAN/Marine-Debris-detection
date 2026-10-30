import cv2
import numpy as np
from pathlib import Path

# Check if any image in unprocessed_data/2021 matches cylider2 visually
figshare_2021 = Path('d:/MarineDebris/unprocessed_data/2021')
cyl_base = Path('d:/MarineDebris/unprocessed_data/cylinder')

cyl_imgs = list(cyl_base.glob('*/**/images/*.jpg'))
fig_imgs = list(figshare_2021.glob('*.jpg'))

print(f'Checking {len(fig_imgs)} Figshare 2021 images against {len(cyl_imgs)} cylider2 images...')

# We can compute downscaled 32x32 grayscale difference
cyl_thumbs = []
for p in cyl_imgs:
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is not None:
        th = cv2.resize(im, (32, 32)).astype(float)
        th -= th.mean()
        norm = np.linalg.norm(th)
        if norm > 0: th /= norm
        cyl_thumbs.append((p, th))

potential_matches = []
for fp in fig_imgs:
    fim = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
    if fim is None: continue
    fth = cv2.resize(fim, (32, 32)).astype(float)
    fth -= fth.mean()
    fnorm = np.linalg.norm(fth)
    if fnorm > 0: fth /= fnorm
    
    for cp, cth in cyl_thumbs:
        sim = np.sum(fth * cth)
        if sim > 0.90:
            potential_matches.append((fp.name, cp.name, sim))

print(f'Potential visual matches found (cosine similarity > 0.90): {len(potential_matches)}')
for m in potential_matches[:5]:
    print(' ', m)
