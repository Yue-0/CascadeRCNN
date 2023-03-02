import sys
from os.path import join
from yaml import safe_load as load

import numpy as np
import paddle as pd

sys.path.append(join(__file__, ".."))

import preprocess

__all__ = ["CascadeRCNN"]

pd.enable_static()
PREPROCESS = {name: eval(f"preprocess.{name}") for name in preprocess.__all__}


class CascadeRCNN:
    """
    Cascade R-CNN detector.
    """
    def __init__(self, model_dir: str, threshold: float = 0.5):
        """
        :param model_dir: Folder where the model is stored.
        :param threshold: Detection threshold, must be between 0 and 1.
        """
        if not 0 < threshold < 1:
            raise ValueError("Param threshold must be between 0 and 1.")
        self.config = pd.inference.Config(*map(
            lambda file: join(model_dir, file),
            (".pd".join(("model",) * 2), ".pdi".join(("model", "params")))
        ))
        with open(join(model_dir, "infer_cfg.yml")) as yaml:
            self.preprocessors = [PREPROCESS[info.pop("type")](**info)
                                  for info in load(yaml)["Preprocess"]]
        self.threshold = threshold
        self.config.enable_use_gpu(200)
        self.config.disable_glog_info()
        self.config.switch_ir_optim(False)
        self.config.enable_memory_optim()
        self.config.switch_use_feed_fetch_ops(False)
        self.predictor = pd.inference.create_predictor(self.config)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        :param image: RGB image for cv2.
        :return: numpy.ndarray[N, 6].
                 # N: number of boxes;
                 # 6: [class, confidence, x_min, y_min, x_max, y_max].
        """
        self.preprocess(image)
        result = self.process()
        return self.reprocess(result)

    def process(self) -> dict:
        self.predictor.run()
        names = self.predictor.get_output_names()
        boxes = self.predictor.get_output_handle(names[0]).copy_to_cpu()
        boxes_num = self.predictor.get_output_handle(names[1]).copy_to_cpu()
        return {"boxes": boxes, "boxes_num": boxes_num}

    def reprocess(self, result: dict) -> dict:
        if result["boxes_num"][0] <= 0:
            np.zeros((0, 6))
        return result["boxes"][
            np.where(result["boxes"][:, 1] >= self.threshold)
        ]

    def preprocess(self, image: np.ndarray) -> None:
        info = {
            "im_shape": np.float32(image.shape[:2]),
            "scale_factor": np.ones(2, np.float32)
        }
        for preprocessor in self.preprocessors:
            image, info = preprocessor(image, info)
        inputs = {
            "image": np.float32((image,)),
            "im_shape": np.float32((info["im_shape"],)),
            "scale_factor": np.float32((info["scale_factor"],))
        }
        for name in self.predictor.get_input_names():
            self.predictor.get_input_handle(name).copy_from_cpu(inputs[name])
