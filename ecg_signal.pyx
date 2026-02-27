# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
import cv2

def image_to_1d_signal(str img_path, int target_len=256):
    cdef:
        np.ndarray[np.uint8_t, ndim=2] img
        np.ndarray[np.uint8_t, ndim=2] binary
        np.ndarray[np.float32_t, ndim=1] signal
        np.ndarray[np.float32_t, ndim=1] out
        np.ndarray[np.int32_t, ndim=1] idx
        int h, w, x, y
        float s, cnt

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found")

    _, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    h = binary.shape[0]
    w = binary.shape[1]
    signal = np.zeros(w, dtype=np.float32)
    for x in range(w):
        s = 0.0
        cnt = 0.0
        for y in range(h):
            if binary[y, x] > 0:
                s += y
                cnt += 1
        if cnt > 0:
            signal[x] = h - (s / cnt)
        else:
            signal[x] = -1.0

    # interpolate missing values
    idx = np.arange(w, dtype=np.int32)
    mask = signal >= 0
    signal[~mask] = np.interp(
        idx[~mask],
        idx[mask],
        signal[mask]
    )

    # normalize
    signal -= signal.mean()
    signal /= (signal.std() + 1e-8)

    out = np.interp(
        np.linspace(0, w - 1, target_len, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        signal
    ).astype(np.float32)

    return out.reshape(1, target_len, 1)
