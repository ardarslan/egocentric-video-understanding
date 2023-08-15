def construct_gen_item(
    dataset,
    video_id,
    frame_idx,
    frame_id,
    image=None,
    delta_images=None,
    original_frame_idx=None,
    activity_verb=None,
    activity_noun=None,
):
    return {
        "dataset": dataset,
        "video_id": video_id,
        "frame_idx": frame_idx,
        "frame_id": frame_id,
        "image": image,
        "delta_images": delta_images,
        "original_frame_idx": original_frame_idx,
        "activity_verb": activity_verb,
        "activity_noun": activity_noun,
    }
