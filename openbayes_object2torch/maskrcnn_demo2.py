from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN
from mmdet.models.dense_heads import RPNHead
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool
import torch.nn as nn
import torch
import numpy as np
from util import ConfigDict
import cv2
from mmdet.datasets import replace_ImageToTensor


class MaskRcnn(nn.Module):
      def __init__(self):
            super(MaskRcnn, self).__init__()

            # build backbone
            self.backbone = ResNet(depth=50,
                              num_stages=4,
                              out_indices=(0, 1, 2, 3),
                              frozen_stages=1,
                              norm_cfg=dict(type='BN', requires_grad=True),
                              norm_eval=True,
                              style='pytorch')
            # build neck
            self.neck = FPN(in_channels=[256, 512, 1024, 2048],
                              out_channels=256,
                              num_outs=5)
            # build rpn_head
            self.rpn_head = RPNHead(in_channels=256,
                              feat_channels=256,
                              anchor_generator=dict(
                                    type='AnchorGenerator',
                                    scales=[8],
                                    ratios=[0.5, 1.0, 2.0],
                                    strides=[4, 8, 16, 32, 64]),
                              bbox_coder=dict(
                                    type='DeltaXYWHBBoxCoder',
                                    target_means=[.0, .0, .0, .0],
                                    target_stds=[1.0, 1.0, 1.0, 1.0]),
                              loss_cls=dict(
                                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                              loss_bbox=dict(type='L1Loss', loss_weight=1.0),
                              test_cfg=ConfigDict(
                                    nms_pre=1000,
                                    max_per_img=1000,
                                    nms=dict(type='nms', iou_threshold=0.7),
                                    min_bbox_size=0))
            # build roi_head 
            self.roi_head = StandardRoIHead(
                              bbox_roi_extractor=dict(
                                    type='SingleRoIExtractor',
                                    roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                                    out_channels=256,
                                    featmap_strides=[4, 8, 16, 32]),
                              bbox_head=dict(
                                    type='Shared2FCBBoxHead',
                                    in_channels=256,
                                    fc_out_channels=1024,
                                    roi_feat_size=7,
                                    num_classes=80,
                                    bbox_coder=dict(
                                    type='DeltaXYWHBBoxCoder',
                                    target_means=[0., 0., 0., 0.],
                                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                                    reg_class_agnostic=False,
                                    loss_cls=dict(
                                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                                    loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
                              mask_roi_extractor=dict(
                                    type='SingleRoIExtractor',
                                    roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                                    out_channels=256,
                                    featmap_strides=[4, 8, 16, 32]),
                              mask_head=dict(
                                    type='FCNMaskHead',
                                    num_convs=4,
                                    in_channels=256,
                                    conv_out_channels=256,
                                    num_classes=80,
                                    loss_mask=dict(
                                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
                              test_cfg=ConfigDict(
                                    score_thr=0.05,
                                    nms=dict(type='nms', iou_threshold=0.5),
                                    max_per_img=100,
                                    mask_thr_binary=0.5))

            self.config_file = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
            self.checkpoint_file = './mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
            self.old_device = 'cpu'
            # self.old_device = 'cuda:0'
            self.old_model = init_detector(self.config_file, self.checkpoint_file, device=self.old_device)
            # load model layers weights,size and names
            self.old_parm = self.get_parm(self.old_model)
            self.old_res = inference_detector(self.old_model, './111.png')
            im = self.old_model.show_result('./111.png', self.old_res)
            cv2.imwrite('old_res.jpg', im)
            self.parm_replace()

      

      def parm_replace(self):
            with torch.no_grad():
                  for name, param in self.named_parameters():
                        if name in self.old_parm.keys():
                              param.copy_(self.old_parm[name])

      #load model layers weights,size and names
      def get_parm(self, model):
            parm={}
            for name,parameters in model.named_parameters():
                  # parm[name]=(parameters.size(), parameters.detach().numpy())
                  parm[name] = parameters
            return parm


      def forward(self, img, img_meta):
            x = self.backbone(img)
            x = self.neck(x)
            proposal_list = self.rpn_head.simple_test_rpn(x, img_meta)
            res = self.roi_head.simple_test(x, proposal_list, img_meta, rescale=False)
            return res

      
      def _predict(self, imgs):
            if isinstance(imgs, (list, tuple)):
                  is_batch = True
            else:
                  imgs = [imgs]
                  is_batch = False

            device = next(self.parameters()).device  # model device

            img_norm_cfg = dict(
                              mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
            test_pipeline = Compose([
                              dict(type='LoadImageFromFile'),
                              dict(
                                    type='MultiScaleFlipAug',
                                    img_scale=(1333, 800),
                                    flip=False,
                                    transforms=[
                                          dict(type='Resize', keep_ratio=True),
                                          dict(type='RandomFlip'),
                                          dict(type='Normalize', **img_norm_cfg),
                                          dict(type='Pad', size_divisor=32),
                                          dict(type='DefaultFormatBundle'),
                                          dict(type='Collect', keys=['img']),
                                    ])
                              ])
            datas = []
            for img in imgs:
                  if isinstance(img, np.ndarray):
                        data = dict(img=img)
                  else:
                        data = dict(img_info=dict(filename=img), img_prefix=None)
                  data = test_pipeline(data)
                  datas.append(data)
            data = collate(datas, samples_per_gpu=len(imgs))
            data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
            data['img'] = [img.data[0] for img in data['img']]
            if next(self.parameters()).is_cuda:
                  data = scatter(data, [device])[0]
            else:
                  for m in self.modules():
                        assert not isinstance(
                              m, RoIPool
                  ), 'CPU inference with RoIPool is not supported currently.'
            with torch.no_grad():
                  results = self.forward(data['img'][0], data['img_metas'][0])

            if not is_batch:
                  return results[0]
            else:
                  return results

      def predict(self, imgs, model):
            if isinstance(imgs, (list, tuple)):
                  is_batch = True
            else:
                  imgs = [imgs]
                  is_batch = False

            cfg = model.cfg
            device = next(model.parameters()).device  # model device

            if isinstance(imgs[0], np.ndarray):
                  cfg = cfg.copy()
                  # set loading pipeline type
                  cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
            test_pipeline = Compose(cfg.data.test.pipeline)

            datas = []
            for img in imgs:
                  # prepare data
                  if isinstance(img, np.ndarray):
                        # directly add img
                        data = dict(img=img)
                  else:
                        # add information into dict
                        data = dict(img_info=dict(filename=img), img_prefix=None)
                        # build the data pipeline
                  data = test_pipeline(data)
                  datas.append(data)

            data = collate(datas, samples_per_gpu=len(imgs))
            # just get the actual data from DataContainer
            data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
            data['img'] = [img.data[0] for img in data['img']]
            if next(model.parameters()).is_cuda:
                  # scatter to specified GPU
                  data = scatter(data, [device])[0]
            else:
                  for m in model.modules():
                        assert not isinstance(
                              m, RoIPool
                  ), 'CPU inference with RoIPool is not supported currently.'

            # forward the model
            with torch.no_grad():
                  # results = model(return_loss=False, rescale=True, **data)
                  results = self.forward(data['img'][0], data['img_metas'][0])

            if not is_batch:
                  return results[0]
            else:
                  return results


if __name__ == '__main__':
      mr = MaskRcnn()
      res = mr.predict('./111.png', mr.old_model)

      print(mr.old_res)
      print('***' * 50)
      print(res)

      im = mr.old_model.show_result('./111.png', res)
      # im = mr.show_result('./111.png', res)
      cv2.imwrite('new_res.jpg', im)





