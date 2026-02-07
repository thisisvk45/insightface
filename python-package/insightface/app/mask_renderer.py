import os
import numpy as np
import os.path as osp
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

from .face_analysis import FaceAnalysis
from ..utils import get_model_dir, DEFAULT_MP_NAME
from ..thirdparty import face3d
from ..data import get_image as ins_get_image


class MaskRenderer:
    def __init__(self, name=DEFAULT_MP_NAME, root="~/.insightface", insfa=None):
        self.mp_name = name
        self.root = root
        self.insfa = insfa

        model_dir = get_model_dir(name, root)

        bfm_file = osp.join(model_dir, "BFM.mat")
        assert osp.exists(bfm_file)
        self.bfm = face3d.morphable_model.MorphabelModel(bfm_file)

        bfm_uv_file = osp.join(model_dir, "BFM_UV.mat")
        assert osp.exists(bfm_uv_file)
        uv_coords = face3d.morphable_model.load.load_uv_coords(bfm_uv_file)

        self.uv_size = (224, 224)
        self.tex_h, self.tex_w = self.uv_size[1], self.uv_size[0]

        texcoord = np.zeros((uv_coords.shape[0], 3), dtype=np.float32)
        texcoord[:, 0] = uv_coords[:, 0] * (self.tex_h - 1)
        texcoord[:, 1] = (1 - uv_coords[:, 1]) * (self.tex_w - 1)
        self.texcoord = texcoord

        self.X_ind = self.bfm.kpt_ind

    def prepare(self, ctx_id=0, det_thresh=0.5, det_size=(128, 128)):
        self._ctx = ctx_id
        self._det_thresh = det_thresh
        self._det_size = det_size

    def _ensure_insfa(self):
        if self.insfa is None:
            self.insfa = FaceAnalysis(
                name=self.mp_name,
                root=self.root,
                allowed_modules=["detection", "landmark_3d_68"],
            )
            self.insfa.prepare(
                ctx_id=self._ctx,
                det_thresh=self._det_thresh,
                det_size=self._det_size,
            )

    def build_params(self, image):
        self._ensure_insfa()
        faces = self.insfa.get(image, max_num=1)
        if not faces:
            return None

        landmark = faces[0].landmark_3d_68[:, :2]
        return self.bfm.fit(landmark, self.X_ind, max_iter=3)

    def params_to_vertices(self, params, h, w):
        sp, ep, s, angles, t = params
        vertices = self.bfm.generate_vertices(sp, ep)
        vertices = self.bfm.transform(vertices, s, angles, t)
        vertices = face3d.mesh.transform.to_image(vertices, h, w)
        return vertices

    def generate_mask_uv(self, mask, pos):
        h, w = self.uv_size[1], self.uv_size[0]
        uv = np.zeros((h, w, 3), dtype=np.uint8)

        stx, sty = int(w * pos[0]), int(h * pos[1])
        etx, ety = int(w * pos[2]), int(h * pos[3])

        mask = cv2.resize(mask, (etx - stx, ety - sty))
        uv[sty:ety, stx:etx] = mask
        return uv

    def render_mask(
        self,
        image,
        mask,
        params,
        input_is_rgb=False,
        auto_blend=True,
        positions=(0.1, 0.33, 0.9, 0.7),
    ):
        if isinstance(mask, str):
            mask = ins_get_image(mask, to_rgb=input_is_rgb)

        uv_mask = self.generate_mask_uv(mask, positions)
        h, w = image.shape[:2]

        verts = self.params_to_vertices(params, h, w)
        rendered = face3d.mesh.render.render_texture(
            verts,
            self.bfm.full_triangles,
            uv_mask,
            self.texcoord,
            self.bfm.full_triangles,
            h,
            w,
        )

        output = ((1 - rendered) * 255).astype(np.uint8)

        if not auto_blend:
            return output

        mask_bin = (output == 255).astype(np.uint8)
        return image * mask_bin + output * (1 - mask_bin)

    @staticmethod
    def encode_params(params):
        return (
            list(params[0])
            + list(params[1])
            + [float(params[2])]
            + list(params[3])
            + list(params[4])
        )

    @staticmethod
    def decode_params(p):
        sp = np.array(p[:199], np.float32).reshape(-1, 1)
        ep = np.array(p[199:228], np.float32).reshape(-1, 1)
        s = p[228]
        angles = tuple(p[229:232])
        t = np.array(p[232:235], np.float32).reshape(-1, 1)
        return sp, ep, s, angles, t


class MaskAugmentation(ImageOnlyTransform):
    def __init__(
        self,
        mask_names,
        mask_probs,
        h_low=0.33,
        h_high=0.35,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.renderer = MaskRenderer()
        self.mask_names = mask_names
        self.mask_probs = mask_probs
        self.h_low = h_low
        self.h_high = h_high

    def apply(self, image, hlabel, mask_name, h_pos, **params):
        if len(hlabel) == 237:
            if hlabel[1] < 0:
                return image
            hlabel = hlabel[2:]

        params = self.renderer.decode_params(hlabel)
        return self.renderer.render_mask(
            image,
            mask_name,
            params,
            input_is_rgb=True,
            positions=(0.1, h_pos, 0.9, 0.7),
        )

    @property
    def targets_as_params(self):
        return ["image", "hlabel"]

    def get_params_dependent_on_targets(self, params):
        return {
            "hlabel": params["hlabel"],
            "mask_name": np.random.choice(self.mask_names, p=self.mask_probs),
            "h_pos": np.random.uniform(self.h_low, self.h_high),
        }

    def get_transform_init_args_names(self):
        return ("mask_names", "mask_probs", "h_low", "h_high")
