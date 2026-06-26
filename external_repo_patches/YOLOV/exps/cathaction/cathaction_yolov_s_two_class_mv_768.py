import os

import torch.nn as nn

from exps.yolov.yolov_s import Exp as MyExp
from yolox.data.data_augment import TrainTransform, Vid_Val_Transform
from yolox.data.datasets import vid

_DATA = "/home/mingzhang/cathaction/datasets/collision_detection_yolov_mv"
_OUT = "/home/mingzhang/cathaction/outputs/task2/yolov_two_class_mv_768"


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 2  # 0=normal, 1=collision
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = _DATA
        self.train_ann = "cathaction_train.json"
        self.val_name = "valid_phantom"  # decision metric = phantom (video_0)
        self.val_ann = "cathaction_valid_phantom.json"
        self.output_dir = _OUT

        self.input_size = (768, 768)
        self.test_size = (768, 768)
        self.max_epoch = 24  # post-warmup iter is ~0.085s (~10min/epoch); 4ep was undertrained (cls just starting). ~4h.
        self.warmup_epochs = 1
        self.no_aug_epochs = 2
        self.pre_no_aug = 2  # required by vid_trainer aug-schedule logic (yolov_s/agnostic omit it)
        self.eval_interval = 1
        self.print_interval = 20
        self.save_history_ckpt = False
        self.data_num_workers = 4

        self.gmode = True
        self.lmode = False
        self.lframe = 0
        self.lframe_val = 0
        self.gframe = 8
        self.gframe_val = 16
        self.tnum = -1
        self.defualt_p = 30
        self.defualt_pre = 750
        self.test_conf = 0.001
        self.nmsthre = 0.5
        self.fix_bn = True
        self.use_aug = False

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        dataset = vid.OVIS(
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            mode="random",
            lframe=self.lframe,
            gframe=batch_size,
            data_dir=self.data_dir,
            name="train",
            COCO_anno=os.path.join(self.data_dir, self.train_ann),
        )
        return vid.get_trans_loader(batch_size=batch_size, data_num_workers=self.data_num_workers, dataset=dataset)

    def get_eval_loader(self, batch_size, tnum=None, data_num_workers=4, formal=False):
        assert batch_size == self.lframe_val + self.gframe_val
        dataset_val = vid.OVIS(
            data_dir=self.data_dir,
            img_size=self.test_size,
            mode="random",
            COCO_anno=os.path.join(self.data_dir, self.val_ann),
            name=self.val_name,
            lframe=self.lframe_val,
            gframe=self.gframe_val,
            val=True,
            preproc=Vid_Val_Transform(),
        )
        return vid.vid_val_loader(batch_size=batch_size, data_num_workers=data_num_workers, dataset=dataset_val)

    def get_model(self):
        model = super().get_model()
        for parameter in model.parameters():
            parameter.requires_grad = True
        if self.fix_bn:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return model
