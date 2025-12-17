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
    """
    # Case 1: original API (path)
    if isinstance(image_or_path, str):
        return estimator.process_one_image(image_or_path, bbox_thr=bbox_thr, use_mask=use_mask)

    # Case 2: numpy frame (assumed RGB)
    frame_rgb = image_or_path
    try:
        return estimator.process_one_image(frame_rgb, bbox_thr=bbox_thr, use_mask=use_mask)
    except Exception:
        if tmp_dir is None or frame_idx is None:
            raise

        os.makedirs(tmp_dir, exist_ok=True)

        # estimator wants a file path -> write temp image (BGR for cv2.imwrite)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        tmp_path = os.path.join(tmp_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(tmp_path, frame_bgr)

        return estimator.process_one_image(tmp_path, bbox_thr=bbox_thr, use_mask=use_mask)


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

        img_bgr = cv2.imread(image_path)
        rend = visualize_sample_together(img_bgr, outputs, estimator.faces)
        rend = np.ascontiguousarray(_to_uint8(rend))

        # If visualize returns RGB, convert to BGR before saving
        if not args.vis_returns_bgr:
            rend = cv2.cvtColor(rend, cv2.COLOR_RGB2BGR)

        out_name = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, rend)


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

    try:
        for idx, frame_bgr, fps, (w, h), n_frames in iter_video_frames_bgr(args.video_path):
            if pbar is None:
                total = n_frames if n_frames and n_frames > 0 else None
                pbar = tqdm(total=total, desc="Processing video frames")

            if idx % args.stride != 0:
                pbar.update(1)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            outputs = process_one_input(
                estimator,
                frame_rgb,
                bbox_thr=args.bbox_thresh,
                use_mask=args.use_mask,
                tmp_dir=tmp_dir,
                frame_idx=idx,
            )

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

            # Enforce constant output size for the entire video (VideoWriter requirement)
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
    Timestamp mode: extract specific frames only and save per-timestamp renders + inputs.
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
  # folder of images
  python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

  # video (process all frames)
  python demo.py --video_path ./video.mp4 --checkpoint_path ./checkpoints/model.ckpt

  # video + timestamps (extract specific frames only)
  python demo.py --video_path ./video.mp4 --video_timestamps --timestamps "01:37.409,01:38.357" --checkpoint_path ./checkpoints/model.ckpt
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

    parser.add_argument("--bbox_thresh", default=0.8, type=float, help="Bounding box detection threshold")
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

    # Timestamp mode options
    parser.add_argument("--timestamps", type=str, default="", help='Comma-separated timestamps e.g. "01:37.409,01:38.357"')
    parser.add_argument("--timestamp_file", type=str, default="", help="Text file with one timestamp per line")

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
