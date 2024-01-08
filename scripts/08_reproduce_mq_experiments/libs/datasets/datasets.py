import os
import torch
from .data_utils import trivial_batch_collator, worker_init_reset_seed
from torch.utils.data.dataloader import default_collate

datasets = {}


def register_dataset(name):
    def decorator(cls):
        datasets[name] = cls
        return cls

    return decorator


def make_dataset(name, is_training, split, **kwargs):
    """
    A simple dataset builder
    """
    dataset = datasets[name](is_training, split, **kwargs)
    return dataset


def make_data_loader(
    dataset,
    is_training,
    generator,
    batch_size,
    num_workers,
    collate_fn="trivial_batch_collator",
):
    """
    A simple dataloder builder
    """
    if collate_fn == "trivial_batch_collator":
        collate_fn = trivial_batch_collator
    elif collate_fn == "default_collator":
        collate_fn = default_collate
    else:
        raise Exception("Not a valid collate fn.")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        persistent_workers=False,
        pin_memory=True,
    )
    return loader
