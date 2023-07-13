# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
# Modified by Xingyi Zhou: reset metadata.thing_classes using loaded label space
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import json
from detectron2.config import get_cfg

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import CfgNode as CN


def add_unidet_config(cfg):
    _C = cfg

    _C.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    _C.MODEL.ROI_BOX_HEAD.PRIOR_PROB = 0.01
    _C.MODEL.ROI_BOX_HEAD.USE_EQL_LOSS = False # Equalization loss described in https://arxiv.org/abs/2003.05176
    _C.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = \
        'datasets/oid/annotations/openimages_challenge_2019_train_v2_cat_info.json'
    _C.MODEL.ROI_BOX_HEAD.EQL_FREQ_CAT = 200
    _C.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False # Federated loss described in https://www.lvisdataset.org/assets/challenge_reports/2020/CenterNet2.pdf, not used in this project
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT = 50
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT = 0.5
    _C.MODEL.ROI_BOX_HEAD.HIERARCHY_PATH = \
        'datasets/oid/annotations/challenge-2019-label500-hierarchy-list.json' # Hierarchical-loss for OpenImages
    _C.MODEL.ROI_BOX_HEAD.HIERARCHY_IGNORE = False # Ignore child classes when the annotation is an internal class
    _C.MODEL.ROI_BOX_HEAD.HIERARCHY_POS_PARENTS = False # Set parent classes in the hierarchical tree as positive 
    _C.MODEL.ROI_BOX_HEAD.UNIFIED_MAP_BACK = True # Ignore out-of-dataset classes for retraining unified model
    _C.MODEL.ROI_BOX_HEAD.FIX_NORM_REG = False # not used
    
    # ResNeSt
    _C.MODEL.RESNETS.DEEP_STEM = False
    _C.MODEL.RESNETS.AVD = False
    _C.MODEL.RESNETS.AVG_DOWN = False
    _C.MODEL.RESNETS.RADIX = 1
    _C.MODEL.RESNETS.BOTTLENECK_WIDTH = 64

    _C.MULTI_DATASET = CN()
    _C.MULTI_DATASET.ENABLED = False
    _C.MULTI_DATASET.DATASETS = ['objects365', 'coco', 'oid']
    _C.MULTI_DATASET.NUM_CLASSES = [365, 80, 500]
    _C.MULTI_DATASET.DATA_RATIO = [1, 1, 1]
    _C.MULTI_DATASET.UNIFIED_LABEL_FILE = ''
    _C.MULTI_DATASET.UNIFY_LABEL_TEST = False # convert the partitioned model to a unified model at test time
    _C.MULTI_DATASET.UNIFIED_EVAL = False
    _C.MULTI_DATASET.SAMPLE_EPOCH_SIZE = 1600
    _C.MULTI_DATASET.USE_CAS = [False, False, False] # class-aware sampling
    _C.MULTI_DATASET.CAS_LAMBDA = 1. # Class aware sampling weight from https://arxiv.org/abs/2005.08455, not used in this project
    _C.MULTI_DATASET.UNIFIED_NOVEL_CLASSES_EVAL = False # zero-shot cross dataset evaluation
    _C.MULTI_DATASET.MATCH_NOVEL_CLASSES_FILE = ''

    _C.SOLVER.RESET_ITER = False # used when loading a checkpoint for finetuning
    _C.CPU_POST_PROCESS = False
    _C.TEST.AUG.NMS_TH = 0.7
    _C.DEBUG = False
    _C.VIS_THRESH = 0.3
    _C.DUMP_CLS_SCORE = False # dump prediction logits to disk. Used for the distortion metric for learning a label space
    _C.DUMP_BBOX = False
    _C.DUMP_NUM_IMG = 2000
    _C.DUMP_NUM_PER_IMG = 50


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_unidet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


class UnifiedVisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get("__unused")
        unified_label_file = json.load(open(cfg.MULTI_DATASET.UNIFIED_LABEL_FILE))
        self.metadata.thing_classes = [
            '{}'.format([xx for xx in x['name'].split('_') if xx != ''][0]) \
                for x in unified_label_file['categories']]
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    visualization = process_predictions(frame, predictions)
                    yield (frame, predictions, visualization)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                visualization = process_predictions(frame, predictions)
                yield (frame, predictions, visualization)
        else:
            for frame in frame_gen:
                predictions = self.predictor(frame)
                visualization = process_predictions(frame, predictions)
                yield (frame, predictions, visualization)


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
