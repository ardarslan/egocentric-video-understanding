import os
import argparse
import torch
from transformers import Blip2Processor
from pytorchvideo.data.video import VideoPathHandler
from video_blip.model import VideoBlipForConditionalGeneration, process


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.path.join(
            os.environ["SCRATCH"], "mq_libs/video-blip-opt-2.7b-ego4d"
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    processor = Blip2Processor.from_pretrained(args.model)
    model = VideoBlipForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to(args.device)

    video_path_handler = VideoPathHandler()

    video_path = os.path.join(
        os.environ["SCRATCH"],
        "ego4d_data/v2/clips",
        "ffe2261f-b973-4fbd-8824-06f8334afdc5.mp4",
    )
    clip = video_path_handler.video_from_path(video_path).get_clip(0, 0.3)

    # sample a frame every 30 frames, i.e. 1 fps. We assume the video is 30 fps for now.
    frames = clip["video"].unsqueeze(0)

    # construct chat context
    text = "Question: What is the camera wearer doing? Answer:"

    # process the inputs
    inputs = process(processor, video=frames, text=text).to(model.device)
    generated_ids = model.generate(
        **inputs, num_beams=4, max_new_tokens=128, temperature=0.7
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()
    print(generated_text)
