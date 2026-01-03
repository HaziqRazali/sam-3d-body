#!/usr/bin/env python3
import argparse
import os
import time
import cv2
import shutil
import tempfile


def parse_args():
    p = argparse.ArgumentParser("Record webcam video, then trim and overwrite")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--subject", default="haziq")
    p.add_argument("--camera-name", default="laptop_webcam")
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--base-dir", default="~/datasets/mocap/data/self")
    p.add_argument("--fourcc", default="mp4v")
    p.add_argument("--no-preview", action="store_true")
    return p.parse_args()


def ensure_size(frame, w, h):
    if frame is None:
        return None
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    return frame


def record_video(out_path, args):
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*args.fourcc),
        args.fps,
        (args.width, args.height),
    )

    print(f"[INFO] Recording → {out_path}")
    print("[INFO] Press 'q' to stop")

    win = "Recording"
    if not args.no_preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = ensure_size(frame, args.width, args.height)
        writer.write(frame)

        if not args.no_preview:
            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if not args.no_preview:
        cv2.destroyAllWindows()


def trim_and_overwrite(video_path, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open recorded video")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start, end = 0, n_frames - 1
    pos = 0
    playing = False

    win = "Trim (i=start, o=end, space=play, s=save, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_trackbar(v):
        nonlocal pos, playing
        pos = v
        playing = False

    cv2.createTrackbar("frame", win, 0, n_frames - 1, on_trackbar)

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        return f if ret else None

    while True:
        pos = max(0, min(pos, n_frames - 1))
        cv2.setTrackbarPos("frame", win, pos)

        frame = read_frame(pos)
        if frame is None:
            break

        overlay = frame.copy()
        cv2.putText(
            overlay,
            f"pos={pos}  start={start}  end={end}  time={pos/fps:.2f}s",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow(win, overlay)

        key = cv2.waitKey(1 if playing else 50) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("i"):
            start = pos
            print(f"[INFO] start={start}")
        elif key == ord("o"):
            end = pos
            print(f"[INFO] end={end}")
        elif key == ord("s"):
            break

        if playing:
            pos += 1
            if pos > end:
                pos = start

    cap.release()
    cv2.destroyAllWindows()

    start, end = min(start, end), max(start, end)
    if end <= start:
        print("[WARN] Invalid trim range, keeping original.")
        return

    # write to temp file first
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    writer = cv2.VideoWriter(
        tmp_path,
        cv2.VideoWriter_fourcc(*args.fourcc),
        fps,
        (w, h),
    )

    cap = cv2.VideoCapture(video_path)
    for i in range(start, end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = ensure_size(frame, w, h)
        writer.write(frame)

    cap.release()
    writer.release()

    shutil.move(tmp_path, video_path)
    print(f"[INFO] Overwritten video saved: {video_path}")


def main():
    args = parse_args()
    args.base_dir = os.path.expanduser(args.base_dir)

    out_dir = os.path.join(
        args.base_dir,
        args.split,
        args.subject,
        "videos",
        args.camera_name,
    )
    os.makedirs(out_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(out_dir, f"{timestamp}.mp4")

    record_video(video_path, args)
    print("[INFO] Opening trim GUI…")
    trim_and_overwrite(video_path, args)


if __name__ == "__main__":
    main()
