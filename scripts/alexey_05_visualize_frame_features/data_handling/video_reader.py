import cv2
import numpy as np
import os
from os.path import join, isdir
from PIL import Image
import time

from utils.globals import *


class VideoReader:
    def __init__(
        self,
        video_path,
        frame_dir_path=None,
        max_width=None,
        max_height=None,
        assumed_fps=-1,
        chunk_size=VIDEO_READER_CHUNK_SIZE,
    ):
        # for compatibility with Python 3.7, do not use {=}
        print(f"VideoReader: video_path={video_path}, frame_dir_path={frame_dir_path}")
        self.video_path = video_path
        self.chunk_cache = {}
        self.frame_dir_path = (
            frame_dir_path
            if frame_dir_path not in ["", None] and isdir(frame_dir_path)
            else None
        )
        self.chunk_size = chunk_size
        self.max_chunk_count = VIDEO_READER_MAX_CHUNK_COUNT
        video_cap = cv2.VideoCapture(video_path)
        self.fps = video_cap.get(cv2.CAP_PROP_FPS)
        self.video_len = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.max_width = max_width
        self.max_height = max_height
        self.assumed_fps = assumed_fps

    def __len__(self):
        return self.video_len

    def get_virtual_frame_count(self):
        num_virtual_frames = int(len(self) / self.fps * self.assumed_fps)
        while int(num_virtual_frames / self.assumed_fps * self.fps) > len(self):
            num_virtual_frames -= 1
        return num_virtual_frames

    def get_real_frame_idx(self, frame_idx):
        return min(self.video_len - 1, round(frame_idx / self.assumed_fps * self.fps))

    def get_virtual_frame_idx(self, frame_idx):
        return min(
            self.get_virtual_frame_count() - 1,
            round(frame_idx / self.fps * self.assumed_fps),
        )

    def get_frame(self, frame, return_real_frame_idx=False):
        if self.assumed_fps != -1:
            orig_frame = frame
            frame = self.get_real_frame_idx(frame)

        if frame < 0:
            print(
                f"WARNING: attempt to read frame {frame} < 0; returning frame 0; orig_frame={orig_frame}"
            )
            frame = 0

        if frame >= self.video_len:
            print(
                f"WARNING: attempt to read frame {frame} >= self.video_len={self.video_len};"
                + f" returning frame {self.video_len-1}; orig_frame={orig_frame}"
            )
            frame = self.video_len - 1

        chunk = frame // self.chunk_size
        if chunk not in self.chunk_cache:
            if len(self.chunk_cache) > self.max_chunk_count:
                min_chunk_key = min(
                    self.chunk_cache.items(), key=lambda el: el[1]["last_access"]
                )[0]
                del self.chunk_cache[min_chunk_key]
            chunk_frames = list(
                range(
                    chunk * self.chunk_size,
                    min((chunk + 1) * self.chunk_size, len(self)),
                )
            )
            self.chunk_cache[chunk] = {
                "last_access": time.time(),
                "frames": self.get_frames(chunk_frames),
            }

        self.chunk_cache[chunk]["last_access"] = time.time()

        ret_frame = self.chunk_cache[chunk]["frames"][frame % self.chunk_size]
        if return_real_frame_idx:
            return ret_frame, frame
        else:
            return ret_frame

    def get_frames(self, frame_idxs):
        missing_img_idxs = {}
        imgs = []
        if self.frame_dir_path not in ["", None]:
            for frame_idx in frame_idxs:
                img_path = join(self.frame_dir_path, f"frame_{frame_idx:07}.jpg")
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
                    img = img[:, :, ::-1]
                    if self.max_width is not None or self.max_height is not None:
                        img_pil = Image.fromarray(img)
                        img_pil.thumbnail(
                            (self.max_width or 1e8, self.max_height or 1e8)
                        )
                        img = np.array(img_pil)
                    imgs.append(img)
                else:
                    missing_img_idxs[frame_idx] = len(imgs)
                    imgs.append(frame_idx)
            if len(missing_img_idxs) == 0:
                return imgs

        video_cap = cv2.VideoCapture(str(self.video_path))
        delta = frame_idxs[1] - frame_idxs[0] if len(frame_idxs) > 1 else 1
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idxs[0])
        video_cap.setExceptionMode(True)

        num_read = 0

        current_frame_idx = frame_idxs[0]
        while num_read <= delta * (len(frame_idxs) - 1):
            success, img = video_cap.read()
            if not success:
                print(
                    f"Error reading from video reader: self.video_path={self.video_path}"
                )
                continue
            if num_read % delta == 0:
                img = img[:, :, ::-1]
                if self.max_width is not None or self.max_height is not None:
                    img_pil = Image.fromarray(img)
                    img_pil.thumbnail((self.max_width or 1e8, self.max_height or 1e8))
                    img = np.array(img_pil)

                if current_frame_idx in missing_img_idxs:
                    imgs[missing_img_idxs[current_frame_idx]] = img
                    del missing_img_idxs[current_frame_idx]
                else:
                    imgs.append(img)
            current_frame_idx += 1
            num_read += 1

        video_cap.release()
        return imgs

    def __getitem__(self, idx):
        return self.get_frame(idx)
