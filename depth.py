import sys
import numpy as np
from PIL import Image
from transformers import pipeline

src, dst = sys.argv[1], sys.argv[2]
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
img = Image.open(src).convert("RGB")
d = np.array(pipe(img)["depth"]).astype(np.float32)
d = (d - d.min()) / (d.max() - d.min() + 1e-8)
Image.fromarray((d * 255).astype(np.uint8)).resize(img.size).save(dst)
print("saved", dst, img.size)
