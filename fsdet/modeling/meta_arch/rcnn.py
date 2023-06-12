# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from torch import nn

from fsdet.structures import ImageList
from fsdet.utils.logger import log_first_n

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY



__all__ = ["GeneralizedRCNN", "ProposalNetwork", "SiameseRCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())  # specify roi_heads name in yaml

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.BACKBONE.FREEZE_P5:
            for connection in [self.backbone.fpn_lateral5, self.backbone.fpn_output5]:
                for p in connection.parameters():
                    p.requires_grad = False
            print('frozen P5 in FPN')


        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC2:
                for p in self.roi_heads.box_head.fc2.parameters():
                    p.requires_grad = True
                print('unfreeze fc2 in roi head')

            # we do not ever need to use this in our works.
            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC1:
                for p in self.roi_heads.box_head.fc1.parameters():
                    p.requires_grad = True
                print('unfreeze fc1 in roi head')
        print('-------- Using Roi Head: {}---------\n'.format(cfg.MODEL.ROI_HEADS.NAME))


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """

        if not self.training:
            return self.inference(batched_inputs)

        # backbone FPN
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)  # List of L, FPN features

        # RPN
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]  # List of N
        else:
            gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # RoI
        # ROI inputs are post_nms_top_k proposals.
        # detector_losses includes Contrast Loss, 和业务层的 cls loss, and reg loss
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # img = Image.open(batched_inputs[0]['file_name'])
        # # img = transforms.ToPILImage()(images.tensor[0].cpu())
        # img = img.resize((800, 800), Image.BILINEAR)
        #
        # for i in range(256):
        #     feat01 = features['p2'][0].cpu()
        #     # heat_pca = torch.pca_lowrank
        #     heat = feat01[i:i+1,:,:]
        #     heat = torch.sigmoid_(heat)
        #     # heat = feat01
        #     heatmap = heat.permute(1, 2, 0)
        #     heatmap = np.uint8(heatmap*255)
        #     heatmap = cv2.resize(heatmap, (800, 800))
        #     # heatmap = cv2.resize(heatmap, (800, 800))
        #
        #     # # heatmap.show()
        #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #     superimposed_img = heatmap * 0.4 + img
        #     file_name = batched_inputs[0]['image_id']
        #     cv2.imwrite('/home/chen/Desktop/FSCE/' + file_name + '_' + str(i)+'.jpg', superimposed_img)

        # feat01 = features['p2'][0].cpu()
        # # heat_pca = torch.pca_lowrank
        # heat = feat01[239:240, :, :]
        # heat = torch.sigmoid_(heat)
        # # heat = feat01
        # heatmap = heat.permute(1, 2, 0)
        # heatmap = np.uint8(heatmap * 255)
        # heatmap = cv2.resize(heatmap, (800, 800))
        # # heatmap = cv2.resize(heatmap, (800, 800))
        #
        # # # heatmap.show()
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # superimposed_img = heatmap * 0.4 + img
        # file_name = batched_inputs[0]['image_id']
        # cv2.imwrite('/home/chen/Desktop/FSCE/' + file_name + '.jpg', superimposed_img)

        if detected_instances is None:
            # print('detected_instances is')
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


@META_ARCH_REGISTRY.register()
class SiameseRCNN(nn.Module):
    """
    Siamese RCNN. Construct MSOCL according to the structure of Siamese networks
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())  # specify roi_heads name in yaml

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.BACKBONE.FREEZE_P5:
            for connection in [self.backbone.fpn_lateral5, self.backbone.fpn_output5]:
                for p in connection.parameters():
                    p.requires_grad = False
            print('frozen P5 in FPN')


        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC2:
                for p in self.roi_heads.box_head.fc2.parameters():
                    p.requires_grad = True
                print('unfreeze fc2 in roi head')

            # we do not ever need to use this in our works.
            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC1:
                for p in self.roi_heads.box_head.fc1.parameters():
                    p.requires_grad = True
                print('unfreeze fc1 in roi head')
        print('-------- Using Roi Head: {}---------\n'.format(cfg.MODEL.ROI_HEADS.NAME))


    def forward(self, DATA):
        """
        Args:
            DATA: a dict, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        # batched_inputs = DATA['data']
        if not self.training:
            return self.inference(DATA)

        # backbone FPN
        batched_inputs = DATA['data']
        images = self.preprocess_image(batched_inputs)

        #　可视化特征图
        # a = images.tensor[2].shape
        # # print(a)
        # img = transforms.ToPILImage()(images.tensor[2].cpu())
        # img.show()
        # a = images.tensor

        features = self.backbone(images.tensor)  # List of L, FPN features

        # 裁剪目标分支数据处理
        if self.training:
            images_crop = self.preprocess_image_crop(DATA['data_crop'])
            features_crop = self.backbone(images_crop.tensor)
        else:
            features_crop = None


        # RPN
        if "instances" in DATA['data'][0]:
            gt_instances = [x["instances"].to(self.device) for x in DATA['data']]  # List of N
            # crop_cls = cat([gt.gt_classes for gt in gt_instances], dim=0)
        else:
            gt_instances = None
            # crop_cls = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
        else:
            assert "proposals" in DATA['data'][0]
            proposals = [x["proposals"].to(self.device) for x in DATA['data']]
            proposal_losses = {}


        # ROI inputs are post_nms_top_k proposals.
        # detector_losses includes Contrast Loss, cls loss, and reg loss
        _, detector_losses = self.roi_heads(images, features, features_crop, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        # 可视化使用
        # images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
                # proposals, _ = self.proposal_generator(batched_inputs, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, None, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
            # for results_per_image, input_per_image, image_size in zip(
            #     results, batched_inputs, batched_inputs.image_sizes
            # ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        # print(batched_inputs)
        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def preprocess_image_crop(self, image_crop):
        """
        Normalize　the cropped branch data, pad and batch the input images.
        """
        images_crop = [x.to(self.device) for x in image_crop]
        images_crop = [self.normalizer(x) for x in images_crop]
        images_crop = ImageList.from_tensors(images_crop, self.backbone.size_divisibility)
        return images_crop


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            # resize the raw outputs of an R-CNN detector to produce outputs according to the desired output resolution.
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
