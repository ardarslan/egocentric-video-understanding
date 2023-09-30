import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, clip_ids, frame_ids):
        "Initialization"
        self.X = X
        self.y = y
        self.clip_ids = clip_ids
        self.frame_ids = frame_ids

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.X)

    def __getitem__(self, idx):
        "Generates one sample of data"
        if self.y is not None:
            return {
                "X": self.X[idx, :],
                "y": self.y[idx],
                "clip_id": self.clip_ids[idx],
                "frame_id": self.frame_ids[idx],
            }
        else:
            return {
                "X": self.X[idx, :],
                "clip_id": self.clip_ids[idx],
                "frame_id": self.frame_ids[idx],
            }
