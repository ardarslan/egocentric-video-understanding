_base_ = [
    "../_base_/models/fcn_hr18.py",
    "../_base_/datasets/egohos_handobj1.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]
model = dict(decode_head=dict(num_classes=6))

checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(
    interval=2000, metric=["mIoU", "mFscore"], pre_eval=True, save_best="mIoU"
)
