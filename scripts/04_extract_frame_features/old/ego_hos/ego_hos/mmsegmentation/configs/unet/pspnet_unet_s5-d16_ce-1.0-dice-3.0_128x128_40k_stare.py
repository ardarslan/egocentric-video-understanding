_base_ = "./pspnet_unet_s5-d16_128x128_40k_stare.py"
model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=1.0),
            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=3.0),
        ]
    )
)
