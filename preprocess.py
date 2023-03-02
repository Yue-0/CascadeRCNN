import cv2
import numpy as np

__all__ = ["Resize", "Permute", "PadStride", "NormalizeImage"]


class Resize:
    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_CUBIC):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.interp = interp
        self.keep_ratio = keep_ratio
        self.target_size = target_size

    def __call__(self, img, info):
        x, y = self._scale(img.shape[:2])
        img = cv2.resize(img, None, None, x, y, self.interp)
        info["im_shape"] = np.float32(img.shape[:2])
        info["scale_factor"] = np.float32([y, x])
        return img, info

    def _scale(self, shape):
        if self.keep_ratio:
            image_size, target_size = tuple(map(
                sorted, (shape, self.target_size)
            ))
            scale = target_size[0] / image_size[0]
            if np.round(scale * image_size[1]) > target_size[1]:
                scale = target_size[1] / image_size[1]
            return scale, scale
        return [self.target_size[s] / shape[s] for s in (1, 0)]


class Permute(object):
    def __call__(self, img, info):
        return img.transpose((2, 0, 1)).copy(), info


class PadStride:
    def __init__(self, stride=32):
        self.stride = stride

    def __call__(self, img, info):
        c, h, w = img.shape
        pad_h = int(np.ceil(float(h) / self.stride) * self.stride)
        pad_w = int(np.ceil(float(w) / self.stride) * self.stride)
        padding = np.zeros((c, pad_h, pad_w), dtype=np.float32)
        padding[:, :h, :w] = img
        return padding, info


class NormalizeImage:
    def __init__(self, mean, std, is_scale=True):
        self.std = std
        self.mean = mean
        self.scale = is_scale

    def __call__(self, img, info):
        img = img.astype(np.float32, copy=False)
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        if self.scale:
            img /= 255.
        return (img - mean) / std, info
