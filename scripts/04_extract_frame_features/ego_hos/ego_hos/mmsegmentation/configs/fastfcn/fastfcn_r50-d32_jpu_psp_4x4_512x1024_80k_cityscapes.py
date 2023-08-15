_base_ = [
    "../_base_/models/fastfcn_r50-d32_jpu_psp.py",
    "../_base_/datasets/cityscapes.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
