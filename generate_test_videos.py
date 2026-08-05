#!/usr/bin/env python3
"""Generate synthetic test videos for CI when no real dataset is provided."""
import cv2
import numpy as np
import os
import sys


def generate(output_dir="data/videos", num_videos=10, frames_per_video=48, width=640, height=360):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_videos):
        out_path = os.path.join(output_dir, f"test_{i:03d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, 24.0, (width, height))

        if not out.isOpened():
            print(f"Failed to open video writer for {out_path}")
            sys.exit(1)

        for f in range(frames_per_video):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            for y in range(height):
                frame[y, :, 0] = int(50 + 100 * (y / height))
                frame[y, :, 1] = int(30 + 80 * (y / height))
                frame[y, :, 2] = int(100 + 155 * (y / height))

            cx = int(width * 0.3 + width * 0.4 * (f / frames_per_video))
            cy = int(height * 0.5 + height * 0.2 * np.sin(f * 0.2 + i))
            cv2.circle(frame, (cx, cy), 30, (255, 200, 100), -1)

            cv2.putText(frame, f"Video {i} Frame {f}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)

        out.release()
        print(f"Generated {out_path}")

    print(f"Done! Generated {num_videos} test videos in {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/videos")
    parser.add_argument("--num_videos", type=int, default=10)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    generate(args.output_dir, args.num_videos, args.frames, args.width, args.height)