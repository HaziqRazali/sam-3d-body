# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
from glob import glob

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample_together
from tqdm import tqdm


def print_output_structure(obj, prefix=""):
    import torch
    import numpy as np

    if isinstance(obj, dict):
        for k, v in obj.items():
            print_output_structure(v, prefix + f"{k}.")
    elif isinstance(obj, (list, tuple)):
        print(prefix + f"[list len={len(obj)}]")
        for i, v in enumerate(obj):
            print_output_structure(v, prefix + f"[{i}].")
    elif isinstance(obj, torch.Tensor):
        print(prefix[:-1], "torch.Tensor", tuple(obj.shape), obj.dtype)
    elif isinstance(obj, np.ndarray):
        print(prefix[:-1], "np.ndarray", obj.shape, obj.dtype)
    elif obj is None:
        print(prefix[:-1], "None")
    else:
        print(prefix[:-1], type(obj).__name__, obj)


# ============================================================
# NEW: choose one person closest to image center
# ============================================================
def select_center_person(outputs, img_w: int, img_h: int, enabled: bool):
    """
    If enabled and outputs contains multiple people (list of dicts),
    return a list with exactly one dict: the person whose bbox center is closest
    to the image center.

    If bbox is missing, fallback to first person.
    """
    if not enabled:
        return outputs

    if not isinstance(outputs, (list, tuple)):
        return outputs

    if len(outputs) <= 1:
        return outputs

    cx_img = img_w * 0.5
    cy_img = img_h * 0.5

    best_i = 0
    best_d2 = None

    for i, out in enumerate(outputs):
        if not isinstance(out, dict):
            continue
        bbox = out.get("bbox", None)
        if bbox is None:
            continue
        bbox = np.asarray(bbox).reshape(-1)
        if bbox.size < 4:
            continue

        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        d2 = (cx - cx_img) ** 2 + (cy - cy_img) ** 2

        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_i = i

    return [outputs[best_i]]


# ============================================================
# NEW: pred_vertices -> vertices aliasing (for all modes)
# ============================================================
def add_vertices_alias(outputs):
    """
    Ensure each person dict has a 'vertices' key aliasing 'pred_vertices'.
    This is a rename/alias convenience: no data copy.
    - If outputs is dict: operate on it.
    - If outputs is list/tuple of dicts: operate per element.
    """
    if isinstance(outputs, dict):
        if ("pred_vertices" in outputs) and ("vertices" not in outputs):
            outputs["vertices"] = outputs["pred_vertices"]
        return outputs

    if isinstance(outputs, (list, tuple)):
        out_list = list(outputs)  # in case it's a tuple
        for o in out_list:
            if isinstance(o, dict) and ("pred_vertices" in o) and ("vertices" not in o):
                o["vertices"] = o["pred_vertices"]
        return out_list

    return outputs


# ============================================================
# Whole-video NPZ (memmap) helpers
# ============================================================
def init_video_memmaps(out_dir, n_kept_frames, dtype=np.float32):
    """
    Disk-backed buffers (memmaps) so we can store the whole video without RAM blow-up.
    Shapes are based on what you printed from outputs[0].
    """
    os.makedirs(out_dir, exist_ok=True)
    mm = {}

    mm["vertices"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "vertices.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 18439, 3),
    )
    mm["pred_pose_raw"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "pred_pose_raw.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 266),
    )
    mm["pred_cam_t"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "pred_cam_t.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 3),
    )
    mm["pred_joint_coords"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "pred_joint_coords.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 127, 3),
    )
    mm["pred_keypoints_3d"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "pred_keypoints_3d.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 70, 3),
    )
    mm["pred_keypoints_2d"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "pred_keypoints_2d.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 70, 2),
    )
    mm["bbox"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "bbox.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 4),
    )
    mm["focal_length"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "focal_length.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames,),
    )
    mm["global_rot"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "global_rot.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 3),
    )
    mm["body_pose_params"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "body_pose_params.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 133),
    )
    mm["hand_pose_params"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "hand_pose_params.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 108),
    )
    mm["scale_params"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "scale_params.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 28),
    )
    mm["shape_params"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "shape_params.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 45),
    )
    mm["expr_params"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "expr_params.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 72),
    )
    mm["pred_global_rots"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "pred_global_rots.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 127, 3, 3),
    )
    # hand bboxes are float64 in your print; we store float32 for consistency
    mm["lhand_bbox"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "lhand_bbox.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 4),
    )
    mm["rhand_bbox"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "rhand_bbox.tmp.npy"),
        mode="w+",
        dtype=dtype,
        shape=(n_kept_frames, 4),
    )
    mm["frame_indices"] = np.lib.format.open_memmap(
        os.path.join(out_dir, "frame_indices.tmp.npy"),
        mode="w+",
        dtype=np.int32,
        shape=(n_kept_frames,),
    )

    return mm


def cleanup_video_memmaps(out_dir):
    """
    Delete temporary .tmp.npy files used for memmaps.
    """
    for f in glob(os.path.join(out_dir, "*.tmp.npy")):
        try:
            os.remove(f)
        except OSError:
            pass


# ============================================================
# Per-sample NPZ helper (image / timestamp)
# ============================================================
def save_single_npz(npz_path, outputs, meta: dict):
    """
    Save one inference result (usually outputs is a list len=1 of dicts) into one NPZ.
    Skips None values.
    """
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 0:
            raise RuntimeError("Empty outputs; cannot save NPZ.")
        out0 = outputs[0]
    elif isinstance(outputs, dict):
        out0 = outputs
    else:
        raise RuntimeError(f"Unexpected outputs type for NPZ saving: {type(outputs)}")

    pack = {}
    for k, v in meta.items():
        pack[f"meta_{k}"] = v

    for k, v in out0.items():
        if v is None:
            continue
        pack[k] = v

    np.savez_compressed(npz_path, **pack)


# ----------------------------
# Video helpers
# ----------------------------
def iter_video_frames_bgr(vid_path):
    """Stream frames as BGR (cv2 default) to avoid loading everything into RAM."""
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {vid_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_val = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    n_frames = int(n_frames_val) if n_frames_val and n_frames_val > 0 else -1

    try:
        idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            yield idx, frame_bgr, fps, (width, height), n_frames
            idx += 1
    finally:
        cap.release()


def extract_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # jump to frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


# ----------------------------
# Timestamp parsing
# ----------------------------
def parse_timestamp_to_seconds(ts: str, fps: float) -> float:
    """
    Supports:
      - "SS.sss" (e.g., "1.32")
      - "MM:SS.sss"
      - "HH:MM:SS.sss"
      - "HH:MM:SS:FF"  (FF = frame number within that second, uses fps)
    """
    ts = ts.strip()
    if not ts:
        raise ValueError("Empty timestamp")

    parts = ts.split(":")
    if len(parts) == 1:
        return float(parts[0])

    if len(parts) == 2:
        mm = int(parts[0])
        ss = float(parts[1])
        return mm * 60.0 + ss

    if len(parts) == 3:
        hh = int(parts[0])
        mm = int(parts[1])
        ss = float(parts[2])
        return hh * 3600.0 + mm * 60.0 + ss

    if len(parts) == 4:
        hh = int(parts[0])
        mm = int(parts[1])
        ss = int(parts[2])
        ff = int(parts[3])
        return hh * 3600.0 + mm * 60.0 + ss + (ff / float(fps))

    raise ValueError(f"Unrecognized timestamp format: {ts}")


def timestamp_to_frame_idx(ts: str, fps: float) -> int:
    sec = parse_timestamp_to_seconds(ts, fps)
    return int(round(sec * fps))


# ----------------------------
# Common helpers
# ----------------------------
def _to_uint8(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def get_output_folder(args):
    if args.output_folder:
        return args.output_folder

    if args.image_folder:
        return os.path.join("./output", os.path.basename(os.path.normpath(args.image_folder)))
    else:
        base = os.path.splitext(os.path.basename(args.video_path))[0]
        return os.path.join("./output", base)


def build_estimator(args):
    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(args.checkpoint_path, device=device, mhr_path=mhr_path)

    # # ✅ PRINT HERE (after model is built+loaded)
    # print(
    #     "B: after load_sam_3d_body, head_pose.faces:",
    #     "sum=", model.head_pose.faces.sum().item(),
    #     "min=", model.head_pose.faces.min().item(),
    #     "max=", model.head_pose.faces.max().item(),
    #     "shape=", tuple(model.head_pose.faces.shape),
    #     "dtype=", model.head_pose.faces.dtype,
    # )

    # # (optional) show which parameter name it is in the model
    # for n, p in model.named_parameters():
    #     if n.endswith("faces") or ".faces" in n:
    #         print("faces param name:", n, "sum=", p.sum().item())
    #         break

    human_detector, human_segmentor, fov_estimator = None, None, None

    if args.detector_name:
        from tools.build_detector import HumanDetector
        human_detector = HumanDetector(name=args.detector_name, device=device, path=detector_path)

    # Keep the newer demo.py behavior:
    # - if segmentor_name != "sam2": build regardless of path
    # - if segmentor_name == "sam2": only build if a path is provided
    if (args.segmentor_name == "sam2" and len(segmentor_path)) or args.segmentor_name != "sam2":
        from tools.build_sam import HumanSegmentor
        human_segmentor = HumanSegmentor(name=args.segmentor_name, device=device, path=segmentor_path)

    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator
        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )
    return estimator


def process_one_input(estimator, image_or_path, bbox_thr, use_mask, tmp_dir=None, frame_idx=None):
    """
    Runs estimator on either:
      - a string path (original behavior)
      - a numpy RGB frame (video frame)
    If estimator only supports paths, fall back to writing a temp jpg (BGR).

    NEW: adds vertices alias (pred_vertices -> vertices) to outputs for consistency.
    """
    # Case 1: original API (path)
    if isinstance(image_or_path, str):
        outputs = estimator.process_one_image(image_or_path, bbox_thr=bbox_thr, use_mask=use_mask)
        return add_vertices_alias(outputs)

    # Case 2: numpy frame (assumed RGB)
    frame_rgb = image_or_path
    try:
        # goes into
        # file:///home/haziq/sam-3d-body/sam_3d_body/sam_3d_body_estimator.py process_one_image()
        outputs = estimator.process_one_image(frame_rgb, bbox_thr=bbox_thr, use_mask=use_mask)

        """
        for k,v in outputs[0].items():
            print(k, v.shape)
        bbox (4,)
        focal_length ()
        pred_keypoints_3d (70, 3)
        pred_keypoints_2d (70, 2)
        pred_vertices (18439, 3)
        pred_cam_t (3,)
        pred_pose_raw (266,)
        global_rot (3,)
        body_pose_params (133,)
        hand_pose_params (108,)
        scale_params (28,)
        shape_params (45,)
        expr_params (72,)
        """
        return add_vertices_alias(outputs)
    
    except Exception:
        if tmp_dir is None or frame_idx is None:
            raise
        
        os.makedirs(tmp_dir, exist_ok=True)

        # estimator wants a file path -> write temp image (BGR for cv2.imwrite)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        tmp_path = os.path.join(tmp_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(tmp_path, frame_bgr)

        outputs = estimator.process_one_image(tmp_path, bbox_thr=bbox_thr, use_mask=use_mask)
        return add_vertices_alias(outputs)


# ----------------------------
# Modes
# ----------------------------
def run_on_image_folder(args, estimator, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.webp"]
    images_list = sorted([img for ext in image_extensions for img in glob(os.path.join(args.image_folder, ext))])

    for image_path in tqdm(images_list, desc="Processing images"):
        outputs = process_one_input(
            estimator,
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
        )

        # NEW: if multiple persons, keep only center person
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        H, W = img_bgr.shape[0], img_bgr.shape[1]
        outputs = select_center_person(outputs, img_w=W, img_h=H, enabled=args.center_person_only)

        rend = visualize_sample_together(img_bgr, outputs, estimator.faces)
        rend = np.ascontiguousarray(_to_uint8(rend))

        # If visualize returns RGB, convert to BGR before saving
        if not args.vis_returns_bgr:
            rend = cv2.cvtColor(rend, cv2.COLOR_RGB2BGR)

        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_folder, base + ".jpg")
        cv2.imwrite(out_path, rend)

        # save npz per image (if requested)
        if args.save_npz:
            npz_path = os.path.join(output_folder, base + ".npz")
            meta = {
                "mode": "image_folder",
                "image_path": image_path,
                "center_person_only": bool(args.center_person_only),
            }
            save_single_npz(npz_path, outputs, meta)

def fill_row_with_nan(memmaps, t, idx):
    # float arrays
    for k in [
        "vertices", "pred_pose_raw", "pred_cam_t", "pred_joint_coords",
        "pred_keypoints_3d", "pred_keypoints_2d", "bbox",
        "global_rot", "body_pose_params", "hand_pose_params",
        "scale_params", "shape_params", "expr_params", "pred_global_rots",
        "lhand_bbox", "rhand_bbox",
    ]:
        memmaps[k][t][...] = np.nan

    # scalar float
    memmaps["focal_length"][t] = np.float32(np.nan)

    # int
    memmaps["frame_indices"][t] = np.int32(idx)


def run_on_video(args, estimator, output_folder):
    """
    Full video mode: stream through frames, run inference, and write rendered video.
    NOTE: Output video resolution follows *rendered frame size* returned by visualize_sample_together.
          This avoids squashing when visualize returns concatenated panels (e.g., 4x width).
    """
    os.makedirs(output_folder, exist_ok=True)

    video_base = os.path.splitext(os.path.basename(args.video_path))[0]
    out_video_path = os.path.join(output_folder, f"{video_base}_rendered.mp4")

    tmp_dir = os.path.join(output_folder, "_tmp_frames_for_inference")
    os.makedirs(tmp_dir, exist_ok=True)

    writer = None
    pbar = None
    out_size = None  # (out_w, out_h)
    out_fps = None

    # Whole-video npz saving (memmap-backed)
    memmaps = None
    memmaps_dir = os.path.join(output_folder, "video_npz_tmp")
    kept_t = 0
    n_kept_frames = None
    n_frames_seen = None

    # allow old flag name as alias (if you used it before)
    save_video_npz = bool(args.save_npz or args.save_video_npz)

    try:
        for idx, frame_bgr, fps, (w, h), n_frames in iter_video_frames_bgr(args.video_path):
            if pbar is None:
                total = n_frames if n_frames and n_frames > 0 else None
                pbar = tqdm(total=total, desc="Processing video frames")
                n_frames_seen = n_frames

                # allocate memmaps once we know frame count
                if save_video_npz:
                    if n_frames is None or n_frames <= 0:
                        raise RuntimeError("Video frame count unknown; cannot allocate whole-video NPZ buffers.")
                    n_kept_frames = (n_frames + args.stride - 1) // args.stride
                    memmaps = init_video_memmaps(memmaps_dir, n_kept_frames)

            if idx % args.stride != 0:
                pbar.update(1)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # goes into 
            # /home/haziq/sam-3d-body/demo.py process_one_input()
            # and then into /home/haziq/sam-3d-body/sam_3d_body/sam_3d_body_estimator.py process_one_image()
            outputs = process_one_input(
                estimator,
                frame_rgb,
                bbox_thr=args.bbox_thresh,
                use_mask=args.use_mask,
                tmp_dir=tmp_dir,
                frame_idx=idx,
            )

            # NEW: if multiple persons, keep only center person
            outputs = select_center_person(outputs, img_w=w, img_h=h, enabled=args.center_person_only)

            # write model outputs into memmaps (one row per kept frame)
            if memmaps is not None:
                # If no detection / no person -> fill NaNs and move on
                if (not isinstance(outputs, (list, tuple))) or (len(outputs) == 0) or (not isinstance(outputs[0], dict)):
                    print(f"Frame {kept_t} has NaNs")
                    fill_row_with_nan(memmaps, kept_t, idx)
                    kept_t += 1
                    # optionally log once in awhile
                    # print(f"[WARN] No person at frame {idx}, wrote NaNs")
                    continue

                person0 = outputs[0]

                memmaps["vertices"][kept_t]             = person0["pred_vertices"].astype(np.float32, copy=False)
                memmaps["pred_pose_raw"][kept_t]        = person0["pred_pose_raw"].astype(np.float32, copy=False)
                memmaps["pred_cam_t"][kept_t]           = person0["pred_cam_t"].astype(np.float32, copy=False)
                memmaps["pred_joint_coords"][kept_t]    = person0["pred_joint_coords"].astype(np.float32, copy=False)
                memmaps["pred_keypoints_3d"][kept_t]    = person0["pred_keypoints_3d"].astype(np.float32, copy=False)
                memmaps["pred_keypoints_2d"][kept_t]    = person0["pred_keypoints_2d"].astype(np.float32, copy=False)
                memmaps["bbox"][kept_t]                 = person0["bbox"].astype(np.float32, copy=False)
                memmaps["focal_length"][kept_t]         = np.float32(person0["focal_length"])

                memmaps["global_rot"][kept_t]           = person0["global_rot"].astype(np.float32, copy=False)
                memmaps["body_pose_params"][kept_t]     = person0["body_pose_params"].astype(np.float32, copy=False)
                memmaps["hand_pose_params"][kept_t]     = person0["hand_pose_params"].astype(np.float32, copy=False)
                memmaps["scale_params"][kept_t]         = person0["scale_params"].astype(np.float32, copy=False)
                memmaps["shape_params"][kept_t]         = person0["shape_params"].astype(np.float32, copy=False)
                memmaps["expr_params"][kept_t]          = person0["expr_params"].astype(np.float32, copy=False)
                memmaps["pred_global_rots"][kept_t]     = person0["pred_global_rots"].astype(np.float32, copy=False)

                # hand bbox may still be None; keep it minimal but safe
                memmaps["lhand_bbox"][kept_t] = np.asarray(person0.get("lhand_bbox", [np.nan]*4), dtype=np.float32).reshape(4)
                memmaps["rhand_bbox"][kept_t] = np.asarray(person0.get("rhand_bbox", [np.nan]*4), dtype=np.float32).reshape(4)

                memmaps["frame_indices"][kept_t] = np.int32(idx)
                kept_t += 1

            # visualize on RGB frame
            rend = visualize_sample_together(frame_rgb, outputs, estimator.faces)
            rend = np.ascontiguousarray(_to_uint8(rend))

            if rend.ndim != 3 or rend.shape[2] != 3:
                raise RuntimeError(f"Unexpected render shape: {rend.shape}")

            # Ensure BGR for cv2 VideoWriter
            if not args.vis_returns_bgr:
                rend_bgr = cv2.cvtColor(rend, cv2.COLOR_RGB2BGR)
            else:
                rend_bgr = rend
            rend_bgr = np.ascontiguousarray(_to_uint8(rend_bgr))

            out_w, out_h = int(rend_bgr.shape[1]), int(rend_bgr.shape[0])

            if writer is None:
                out_size = (out_w, out_h)
                out_fps = fps
                fourcc = cv2.VideoWriter_fourcc(*args.video_codec)
                writer = cv2.VideoWriter(out_video_path, fourcc, out_fps, out_size)
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not open VideoWriter at {out_video_path}. "
                        f"Try --video_codec mp4v or avc1 or MJPG."
                    )

                if args.print_debug:
                    print("Input video size:", (w, h))
                    print("Render size:", out_size)
                    print("VideoWriter:", out_video_path)
                    print("FPS:", out_fps, "Codec:", args.video_codec)
                    print("First rendered frame:", rend_bgr.shape, rend_bgr.dtype)

            if out_size != (out_w, out_h):
                raise RuntimeError(
                    f"Rendered frame size changed from initial {out_size} to {(out_w, out_h)} at frame {idx}. "
                    "If your visualization dynamically changes layout, we can add an opt-in resize/crop policy."
                )

            writer.write(rend_bgr)

            if args.save_frames:
                out_frame_path = os.path.join(output_folder, f"frame_{idx:06d}.jpg")
                cv2.imwrite(out_frame_path, rend_bgr)

            pbar.update(1)

    finally:
        if pbar is not None:
            pbar.close()
        if writer is not None:
            writer.release()

    # finalize one single NPZ per video
    if memmaps is not None:
        T = kept_t

        # name
        video_npz_name = args.video_npz_name.strip() if args.video_npz_name else ""
        if video_npz_name == "":
            video_npz_name = f"{video_base}_mhr_outputs.npz"
        if not video_npz_name.endswith(".npz"):
            video_npz_name += ".npz"

        out_npz_path = os.path.join(output_folder, video_npz_name)

        meta = {
            "mode": "video_full",
            "video_path": args.video_path,
            "fps": float(out_fps if out_fps is not None else 0.0),
            "input_w": int(w),
            "input_h": int(h),
            "stride": int(args.stride),
            "num_frames_total": int(n_frames_seen) if n_frames_seen is not None else -1,
            "num_frames_saved": int(T),
            "center_person_only": bool(args.center_person_only),
        }

        for mm in memmaps.values():
            mm.flush()

        np.savez_compressed(
            out_npz_path,
            vertices=memmaps["vertices"][:T],
            pred_pose_raw=memmaps["pred_pose_raw"][:T],
            pred_cam_t=memmaps["pred_cam_t"][:T],
            pred_joint_coords=memmaps["pred_joint_coords"][:T],
            pred_keypoints_3d=memmaps["pred_keypoints_3d"][:T],
            pred_keypoints_2d=memmaps["pred_keypoints_2d"][:T],
            bbox=memmaps["bbox"][:T],
            focal_length=memmaps["focal_length"][:T],
            global_rot=memmaps["global_rot"][:T],
            body_pose_params=memmaps["body_pose_params"][:T],
            hand_pose_params=memmaps["hand_pose_params"][:T],
            scale_params=memmaps["scale_params"][:T],
            shape_params=memmaps["shape_params"][:T],
            expr_params=memmaps["expr_params"][:T],
            pred_global_rots=memmaps["pred_global_rots"][:T],
            lhand_bbox=memmaps["lhand_bbox"][:T],
            rhand_bbox=memmaps["rhand_bbox"][:T],
            frame_indices=memmaps["frame_indices"][:T],
            meta=np.array([meta], dtype=object),
        )

        if not args.keep_video_npz_tmp:
            cleanup_video_memmaps(memmaps_dir)
            try:
                os.rmdir(memmaps_dir)
            except OSError:
                pass

    if args.cleanup_tmp and os.path.isdir(tmp_dir):
        for f in glob(os.path.join(tmp_dir, "*.jpg")):
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


def run_on_video_timestamps(args, estimator, output_folder):
    """
    Timestamp mode: extract specific frames only and save per-timestamp renders + inputs (+ optional npz).
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    cap.release()

    if args.timestamps:
        ts_list = [t.strip() for t in args.timestamps.split(",") if t.strip()]
    else:
        with open(args.timestamp_file, "r") as f:
            ts_list = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(ts_list) == 0:
        raise RuntimeError("No timestamps provided.")

    tmp_dir = os.path.join(output_folder, "_tmp_frames_for_inference")
    os.makedirs(tmp_dir, exist_ok=True)

    for i, ts in enumerate(tqdm(ts_list, desc="Processing timestamps")):
        frame_idx = timestamp_to_frame_idx(ts, fps)
        frame_rgb = extract_frame(args.video_path, frame_idx)

        outputs = process_one_input(
            estimator,
            frame_rgb,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
            tmp_dir=tmp_dir,
            frame_idx=frame_idx,
        )

        # NEW: if multiple persons, keep only center person
        H, W = frame_rgb.shape[0], frame_rgb.shape[1]
        outputs = select_center_person(outputs, img_w=W, img_h=H, enabled=args.center_person_only)

        safe_ts = ts.replace(":", "-")
        base = os.path.join(output_folder, f"ts_{i:04d}_{safe_ts}_f{frame_idx:06d}")

        # 1) raw extracted frame (save as jpg via BGR)
        cv2.imwrite(base + "_input.jpg", cv2.cvtColor(_to_uint8(frame_rgb), cv2.COLOR_RGB2BGR))

        # 2) rendered visualization
        rend = visualize_sample_together(frame_rgb, outputs, estimator.faces)
        rend = np.ascontiguousarray(_to_uint8(rend))

        if not args.vis_returns_bgr:
            rend_bgr = cv2.cvtColor(rend, cv2.COLOR_RGB2BGR)
        else:
            rend_bgr = rend

        cv2.imwrite(base + "_render.jpg", np.ascontiguousarray(_to_uint8(rend_bgr)))

        # save npz per timestamp (if requested)
        if args.save_npz:
            npz_path = base + "_data.npz"
            meta = {
                "mode": "video_timestamps",
                "video_path": args.video_path,
                "timestamp": ts,
                "frame_idx": int(frame_idx),
                "fps": float(fps),
                "center_person_only": bool(args.center_person_only),
            }
            save_single_npz(npz_path, outputs, meta)

    if args.cleanup_tmp and os.path.isdir(tmp_dir):
        for f in glob(os.path.join(tmp_dir, "*.jpg")):
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


def main(args):
    base_output = get_output_folder(args)

    # Keep timestamp outputs in a subfolder to avoid mixing with full-video outputs.
    if args.video_timestamps:
        output_folder = os.path.join(base_output, "timestamps")
    else:
        output_folder = base_output

    os.makedirs(output_folder, exist_ok=True)
    estimator = build_estimator(args)

    if args.image_folder:
        run_on_image_folder(args, estimator, output_folder)
        return

    # video_path mode
    if args.video_timestamps:
        if not args.video_path:
            raise RuntimeError("--video_timestamps requires --video_path")
        if (not args.timestamps) and (not args.timestamp_file):
            raise RuntimeError("--video_timestamps requires --timestamps or --timestamp_file")
        run_on_video_timestamps(args, estimator, output_folder)
    else:
        run_on_video(args, estimator, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Image Folder OR Video Human Mesh Recovery (Full Video or Timestamps)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  # folder of images (keep center person only)
  python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt --save_npz --center_person_only

  # video (process all frames) + save ONE NPZ per video (keep center person only)
  python demo.py --video_path ./video.mp4 --checkpoint_path ./checkpoints/model.ckpt --save_npz --center_person_only

  # video + timestamps (extract specific frames only) + save per-timestamp NPZ (keep center person only)
  python demo.py --video_path ./video.mp4 --video_timestamps --timestamps "01:37.409,01:38.357" --checkpoint_path ./checkpoints/model.ckpt --save_npz --center_person_only
        """,
    )

    # Input source: either image folder OR video path
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--image_folder", type=str, default=None, help="Path to folder containing input images")
    inp.add_argument("--video_path", type=str, default=None, help="Path to an input video file (e.g., .mp4/.avi)")

    # Mode modifier (works only with --video_path)
    parser.add_argument(
        "--video_timestamps",
        action="store_true",
        help="Process specific frames from the video using timestamps (requires --video_path).",
    )

    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name> or ./output/<video_name>)",
    )
    parser.add_argument("--checkpoint_path", required=True, type=str, help="Path to SAM 3D Body model checkpoint")

    parser.add_argument("--detector_name", default="vitdet", type=str, help="Human detection model (Default `vitdet`).")
    parser.add_argument("--segmentor_name", default="sam2", type=str, help="Human segmentation model (Default `sam2`).")
    parser.add_argument("--fov_name", default="moge2", type=str, help="FOV estimation model (Default `moge2`).")

    parser.add_argument("--detector_path", default="", type=str, help="Path to detector model folder (or SAM3D_DETECTOR_PATH)")
    parser.add_argument("--segmentor_path", default="", type=str, help="Path to segmentor model folder (or SAM3D_SEGMENTOR_PATH)")
    parser.add_argument("--fov_path", default="", type=str, help="Path to fov estimation model folder (or SAM3D_FOV_PATH)")
    parser.add_argument("--mhr_path", default="", type=str, help="Path to MoHR/assets folder (or SAM3D_MHR_PATH)")

    parser.add_argument("--bbox_thresh", default=0.7, type=float, help="Bounding box detection threshold")
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask auto-generated from bbox)",
    )

    # Full-video options
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame (default: 1)")
    parser.add_argument("--video_codec", type=str, default="mp4v", help="FourCC codec (common: mp4v, avc1, MJPG)")
    parser.add_argument("--save_frames", action="store_true", default=False, help="Also save rendered frames as JPGs (full-video mode)")
    parser.add_argument("--cleanup_tmp", action="store_true", default=False, help="Delete temporary inference frames")

    # NPZ saving (unified flag)
    parser.add_argument(
        "--save_npz",
        action="store_true",
        default=False,
        help="Save NPZ outputs. Image mode: one NPZ per image. Timestamp mode: one NPZ per timestamp. Full video: ONE NPZ per video (stacked).",
    )

    # Backward-compatible alias (optional): if present, behaves like --save_npz for full video
    parser.add_argument(
        "--save_video_npz",
        action="store_true",
        default=False,
        help="(Alias) Save ONE NPZ per video in full-video mode (stacked arrays). Prefer --save_npz.",
    )

    parser.add_argument(
        "--video_npz_name",
        type=str,
        default="",
        help="Filename for the full-video NPZ (default: <video_basename>_mhr_outputs.npz).",
    )
    parser.add_argument(
        "--keep_video_npz_tmp",
        action="store_true",
        default=False,
        help="Keep temporary memmap .tmp.npy files (debug). By default they are deleted after NPZ is written.",
    )

    # Timestamp mode options
    parser.add_argument("--timestamps", type=str, default="", help='Comma-separated timestamps e.g. "01:37.409,01:38.357"')
    parser.add_argument("--timestamp_file", type=str, default="", help="Text file with one timestamp per line")

    # NEW: single-person selection option
    parser.add_argument(
        "--center_person_only",
        action="store_true",
        default=False,
        help="If multiple people are detected, keep only the person whose bbox center is closest to the image center.",
    )

    # Format handling / debug
    parser.add_argument(
        "--vis_returns_bgr",
        action="store_true",
        default=False,
        help="Set this if visualize_sample_together returns BGR already (then no RGB->BGR conversion is applied).",
    )
    parser.add_argument("--print_debug", action="store_true", default=False, help="Print codec/size debug info on first frame")

    args = parser.parse_args()
    main(args)
