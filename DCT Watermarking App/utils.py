import cv2
import numpy as np

def dct2(block):
    return cv2.dct(np.float32(block))

def idct2(block):
    return cv2.idct(block)

def embed_watermark(image, watermark):
    h, w = image.shape[:2]
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel = image_yuv[:, :, 0]

    y_dct = dct2(y_channel)

    wh, ww = watermark.shape
    y_dct[:wh, :ww] += watermark

    y_idct = idct2(y_dct)
    image_yuv[:, :, 0] = y_idct
    watermarked_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    return watermarked_image

def extract_watermark(image, shape):
    h, w = image.shape[:2]
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel = image_yuv[:, :, 0]

    y_dct = dct2(y_channel)
    watermark = y_dct[:shape[0], :shape[1]]

    return watermark
