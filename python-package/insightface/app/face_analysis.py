# -*- coding: utf-8 -*-
# Optimized FaceAnalysis
# Focus: batching, speed, clean flow

from __future__ import division

import glob
import os.path as osp
import numpy as np
import onnxruntime as ort
from numpy.linalg import norm

from ..model_zoo import model_zoo
from ..utils import DEFAULT_MP_NAME, ensure_available
from .common import Face

__all__ = ["FaceAnalysis"]


class FaceAnalysis:
    def __init__(self, name=DEFAULT_MP_NAME, root="~/.insightface", allowed_modules=None, **kwargs):
        ort.set_default_logger_severity(3)

        self.models = {}
        self.model_dir = ensure_available("models", name, root=root)

        onnx_files = sorted(glob.glob(osp.join(self.model_dir, "*.onnx")))

        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                continue
            if allowed_modules and model.taskname not in allowed_modules:
                continue
            if model.taskname not in self.models:
                self.models[model.taskname] = model

        assert "detection" in self.models
        self.det_model = self.models["detection"]

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        self.det_size = det_size

        for taskname, model in self.models.items():
            if taskname == "detection":
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0, det_metric="default"):
        bboxes, kpss = self.det_model.detect(
            img, max_num=max_num, metric=det_metric
        )

        if bboxes.shape[0] == 0:
            return []

        faces = []
        for i in range(bboxes.shape[0]):
            face = Face(
                bbox=bboxes[i, 0:4],
                det_score=bboxes[i, 4],
                kps=None if kpss is None else kpss[i],
            )
            faces.append(face)

        # batch-friendly models (recognition, age/gender, etc.)
        for taskname, model in self.models.items():
            if taskname == "detection":
                continue
            model.get(img, faces)

        # explicit embedding normalization (important)
        for face in faces:
            if hasattr(face, "embedding") and face.embedding is not None:
                face.embedding = face.embedding / norm(face.embedding)

        return faces

    def draw_on(self, img, faces):
        import cv2

        dimg = img.copy()
        for face in faces:
            box = face.bbox.astype(int)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

            if face.kps is not None:
                kps = face.kps.astype(int)
                for i, (x, y) in enumerate(kps):
                    color = (0, 255, 0) if i in (0, 3) else (0, 0, 255)
                    cv2.circle(dimg, (x, y), 1, color, 2)

            if getattr(face, "gender", None) is not None and getattr(face, "age", None) is not None:
                cv2.putText(
                    dimg,
                    f"{face.sex},{face.age}",
                    (box[0], box[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                )
        return dimg
