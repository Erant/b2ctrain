# Upper-body crops (top 45% of the subject's bbox) of chosen novel renders, one row per variant.
import sys, cv2, numpy as np
from pathlib import Path
work = Path(sys.argv[1]); names = sys.argv[2:]
keys = ["pod", "local162", "n162", "p81", "p81n"]
rows = []
for k in keys:
    tiles = []
    for n in names:
        im = cv2.imread(str(work / k / "novel_white" / n), cv2.IMREAD_UNCHANGED)
        if k == "pod":
            a = im[:, :, 3] > 20; ys, xs = np.nonzero(a)
            y0, y1, x0, x1 = ys.min(), ys.min() + int(0.45 * (ys.max() - ys.min())), xs.min(), xs.max()
            box = (max(y0 - 20, 0), y1, max(x0 - 20, 0), x1 + 20); boxes = getattr(sys, "_b", {}); boxes[n] = box; sys._b = boxes
        y0, y1, x0, x1 = sys._b[n]
        c = im[y0:y1, x0:x1, :3]
        tiles.append(cv2.resize(c, (int(c.shape[1] * 520 / c.shape[0]), 520)))
    row = np.hstack(tiles); cv2.putText(row, k, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2); rows.append(row)
cv2.imwrite(str(work / "arm_crops.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
