_base_ = ["./segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py"]

model = dict(
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint="pretrain/mit_b5.pth"),
        embed_dims=64,
        num_layers=[3, 6, 40, 3],
    ),
    decode_head=dict(in_channels=[64, 128, 320, 512]),
)
