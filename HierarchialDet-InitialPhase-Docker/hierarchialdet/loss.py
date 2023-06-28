# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
import torchvision.ops as ops
from .util import box_ops
from .util.misc import get_world_size, is_dist_avail_and_initialized
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou


class SetCriterionDynamicK(nn.Module):
    """ This class computes the loss for DiffusionDet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses, use_focal):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        if self.use_fed_loss:
            self.fed_loss_num_classes = 4
            #from detectron2.data.detection_utils import get_fed_loss_cls_weights
            from typing import List, Union
            
            def get_fed_loss_cls_weights(dataset_names: Union[str, List[str]], freq_weight_power=1.0,i=0):
                """
                Get frequency weight for each class sorted by class id.
                We now calcualte freqency weight using image_count to the power freq_weight_power.

                Args:
                    dataset_names: list of dataset names
                    freq_weight_power: power value
                """
                if isinstance(dataset_names, str):
                    dataset_names = [dataset_names]

                #check_metadata_consistency("class_image_count", dataset_names)
                

                #meta = MetadataCatalog.get(dataset_names[0])
                #class_freq_meta = meta.class_image_count
                #class_freq = torch.tensor(
                #    [c["image_count"] for c in sorted(class_freq_meta, key=lambda x: x["id"])]
                #)
                class_freq=[]
                
                
                
                
                
                class_freq.append(torch.tensor([247, 195, 3051, 865])) 
               
                   
                
                class_freq.append(torch.tensor([628, 623, 618, 588, 565, 565, 578, 341,]))   
                
                class_freq.append(torch.tensor([693,693,693,693]))   
                
               
                 
                
                
              
                
                
                
                  
               
                
                
                
                
                
                 
               
               
               
                
               
               
            
                
                
                #class_freq=torch.tensor([628, 623, 618, 588, 565, 565, 578, 341, 627, 617, 621, 576, 560, 546, 576, 627, 630, 630, 619, 588, 578, 394, 626, 627, 630, 621, 586, 483, 582, 402, 476, 367,])
                class_freq_weight=[]
                for i in range(len(class_freq)):
                  class_freq_weight.append(class_freq[i].float() ** freq_weight_power)
                
                return class_freq_weight

                
    
            


            cls_weight_fun = lambda: get_fed_loss_cls_weights(dataset_names=cfg.DATASETS.TRAIN, freq_weight_power=cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER)  # noqa

    
            fed_loss_cls_weights = cls_weight_fun()
           
          
            #self.i=self.i+1
            assert (
                    len(fed_loss_cls_weights[0]) == self.num_classes[0]
            ), "Please check the provided fed_loss_cls_weights_1. Their size should match num_classes_1"
            self.register_buffer("fed_loss_cls_weights_1", fed_loss_cls_weights[0])
            assert (
                    len(fed_loss_cls_weights[1]) == self.num_classes[1]
            ), "Please check the provided fed_loss_cls_weights_2. Their size should match num_classes_2"
            self.register_buffer("fed_loss_cls_weights_2", fed_loss_cls_weights[1])
            
            assert (
                    len(fed_loss_cls_weights[2]) == self.num_classes[2]
            ), "Please check the provided fed_loss_cls_weights_3. Their size should match num_classes_3"
            self.register_buffer("fed_loss_cls_weights_3", fed_loss_cls_weights[2])
            
            
           
            
            

        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.DiffusionDet.ALPHA
            self.focal_loss_gamma = cfg.MODEL.DiffusionDet.GAMMA
        else:
            emtpy_weight=[]
            for num_class in self.num_classes:
              empty_weight_inp=torch.ones(self.num_class + 1)
              empty_weight_inp[-1] = self.eos_coef
              empty_weight.append(empty_weight_inp)
            self.register_buffer('empty_weight_1', empty_weight[0])
            self.register_buffer('empty_weight_2', empty_weight[1])
            self.register_buffer('empty_weight_3', empty_weight[2])
    # copy-paste from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L356
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
            Will sample negative classes if number of unique gt_classes is smaller than this value.
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes
        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            import sys
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        else:
            fed_loss_classes = unique_gt_classes
        return fed_loss_classes

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if 'pred_logits_1' in outputs:
          src_logits_1 = outputs['pred_logits_1']
        
        if 'pred_logits_2' in outputs:
          src_logits_2 = outputs['pred_logits_2']
        
        if 'pred_logits_3' in outputs:
          src_logits_3 = outputs['pred_logits_3']
        batch_size = len(targets)

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        
        
        try:
          target_classes_1 = torch.full(src_logits_1.shape[:2], self.num_classes[0],
                                      dtype=torch.int64, device=src_logits_1.device)
          self.freeze_class1=False
        except:
          self.freeze_class1=True
        try:                            
          target_classes_2 = torch.full(src_logits_2.shape[:2], self.num_classes[1],
                                      dtype=torch.int64, device=src_logits_2.device)
          self.freeze_class2=False
        except:
          self.freeze_class2=True
        
        try:
          target_classes_3 = torch.full(src_logits_3.shape[:2], self.num_classes[2],
                                      dtype=torch.int64, device=src_logits_3.device)
          self.freeze_class3=False
        except:
          self.freeze_class3=True
                                    
        
        src_logits_list_1 = []
        src_logits_list_2 = []
        src_logits_list_3 = []
        target_classes_o_1_list = []
        target_classes_o_2_list = []
        target_classes_o_3_list = []
        # target_classes[idx] = target_classes_o
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            
            
            if len(gt_multi_idx) == 0:
                continue
                
            if not self.freeze_class1:
              bz_src_logits_1 = src_logits_1[batch_idx]
              target_classes_o_1 = targets[batch_idx]["labels_1"]
              target_classes_1[batch_idx, valid_query] = target_classes_o_1[gt_multi_idx]
              src_logits_list_1.append(bz_src_logits_1[valid_query])
              target_classes_o_1_list.append(target_classes_o_1[gt_multi_idx])
            
            if not self.freeze_class2:
              bz_src_logits_2 = src_logits_2[batch_idx]
              target_classes_o_2 = targets[batch_idx]["labels_2"]
              target_classes_2[batch_idx, valid_query] = target_classes_o_2[gt_multi_idx]
              src_logits_list_2.append(bz_src_logits_2[valid_query])
              target_classes_o_2_list.append(target_classes_o_2[gt_multi_idx])
            
            if not self.freeze_class3:  
              bz_src_logits_3 = src_logits_3[batch_idx]
              target_classes_o_3 = targets[batch_idx]["labels_3"]
              target_classes_3[batch_idx, valid_query] = target_classes_o_3[gt_multi_idx]
              src_logits_list_3.append(bz_src_logits_1[valid_query])
              target_classes_o_3_list.append(target_classes_o_3[gt_multi_idx])
             
            
            
            
            
            

           
            
            
            
           
           
            

        if self.use_focal or self.use_fed_loss:
            num_boxes = torch.cat(target_classes_o_1_list).shape[0] if len(target_classes_o_1_list) != 0 else 1

            if not self.freeze_class1:
              target_classes_onehot_1 = torch.zeros([src_logits_1.shape[0], src_logits_1.shape[1], self.num_classes[0] + 1],
                                                  dtype=src_logits_1.dtype, layout=src_logits_1.layout,
                                                  device=src_logits_1.device)
              target_classes_onehot_1.scatter_(2, target_classes_1.unsqueeze(-1), 1)                                  
              gt_classes_1 = torch.argmax(target_classes_onehot_1, dim=-1)   
              target_classes_onehot_1 = target_classes_onehot_1[:, :, :-1]    
              src_logits_1 = src_logits_1.flatten(0, 1)                           
              target_classes_onehot_1 = target_classes_onehot_1.flatten(0, 1) 
            
            
            if not self.freeze_class2:                                    
              target_classes_onehot_2 = torch.zeros([src_logits_2.shape[0], src_logits_2.shape[1], self.num_classes[1] + 1],
                                                  dtype=src_logits_2.dtype, layout=src_logits_2.layout,
                                                  device=src_logits_2.device)
              target_classes_onehot_2.scatter_(2, target_classes_2.unsqueeze(-1), 1)                                  
              gt_classes_2 = torch.argmax(target_classes_onehot_2, dim=-1)   
              target_classes_onehot_2 = target_classes_onehot_2[:, :, :-1]    
              src_logits_2 = src_logits_2.flatten(0, 1)                           
              target_classes_onehot_2 = target_classes_onehot_2.flatten(0, 1) 
              
            
            if not self.freeze_class3:                                    
              target_classes_onehot_3 = torch.zeros([src_logits_3.shape[0], src_logits_3.shape[1], self.num_classes[2] + 1],
                                                  dtype=src_logits_3.dtype, layout=src_logits_3.layout,
                                                  device=src_logits_3  .device)
              target_classes_onehot_3.scatter_(2, target_classes_3.unsqueeze(-1), 1)                                  
              gt_classes_3 = torch.argmax(target_classes_onehot_3, dim=-1)   
              target_classes_onehot_3 = target_classes_onehot_3[:, :, :-1]    
              src_logits_3 = src_logits_3.flatten(0, 1)                           
              target_classes_onehot_3 = target_classes_onehot_3.flatten(0, 1)                                 
                                                
            """                                    
            target_classes_onehot_3 = torch.zeros([src_logits_3.shape[0], src_logits_3.shape[1], self.num_classes[2] + 1],
                                                dtype=src_logits_3.dtype, layout=src_logits_3.layout,
                                                device=src_logits_3.device)
            target_classes_onehot_1.scatter_(2, target_classes_1.unsqueeze(-1), 1)
            
            target_classes_onehot_3.scatter_(2, target_classes_3.unsqueeze(-1), 1)
            
            
            gt_classes_1 = torch.argmax(target_classes_onehot_1, dim=-1)
           
            gt_classes_3 = torch.argmax(target_classes_onehot_3, dim=-1)
            
            target_classes_onehot_1 = target_classes_onehot_1[:, :, :-1]
            
            target_classes_onehot_3 = target_classes_onehot_3[:, :, :-1]

            src_logits_1 = src_logits_1.flatten(0, 1)
            
            src_logits_3 = src_logits_3.flatten(0, 1)
            
            target_classes_onehot_1 = target_classes_onehot_1.flatten(0, 1)
         
            target_classes_onehot_3 = target_classes_onehot_3.flatten(0, 1)
            """
            
            if self.use_focal:
                if not self.freeze_class1:
                  cls_loss_1 = sigmoid_focal_loss_jit(src_logits_1, target_classes_onehot_1, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="none")
                if not self.freeze_class2: 
                  cls_loss_2 = sigmoid_focal_loss_jit(src_logits_2, target_classes_onehot_2, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="none")
                if not self.freeze_class3:   
                  cls_loss_3 = sigmoid_focal_loss_jit(src_logits_3, target_classes_onehot_3, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="none")
            else:
                cls_loss_1 = F.binary_cross_entropy_with_logits(src_logits_1, target_classes_onehot_1, reduction="none")
                if not self.freeze_class2: 
                  cls_loss_2= F.binary_cross_entropy_with_logits(src_logits_2, target_classes_onehot_2, reduction="none")
                if not self.freeze_class3:   
                  cls_loss_3 = F.binary_cross_entropy_with_logits(src_logits_3, target_classes_onehot_3, reduction="none")
            if not self.freeze_class2 and not self.freeze_class3:       
              cls_loss=[cls_loss_1, cls_loss_2, cls_loss_3]
              
              
            if not self.freeze_class1 and not self.freeze_class2 and self.freeze_class3:       
              cls_loss=[cls_loss_1, cls_loss_2]
              
              
            if self.freeze_class2 and self.freeze_class3:       
              cls_loss=[cls_loss_1]    
            if self.freeze_class1:
              cls_loss=[cls_loss_2]  
            if self.use_fed_loss:
                K = self.num_classes
                try:
                  N1 = src_logits_1.shape[0]
                except:
                  pass
                try:
                  N2 = src_logits_2.shape[0]
                except:
                  pass
                
                try:  
                  N3 = src_logits_3.shape[0]
                except:
                  pass
                  
                if not self.freeze_class2 and not self.freeze_class3:       
                  N=[N1,N2,N3]
                  
                if not self.freeze_class1 and not self.freeze_class2 and self.freeze_class3:  
                   N=[N1,N2]
                   
                if self.freeze_class2 and self.freeze_class3:  
                   N=[N1]   
                if self.freeze_class1:
                  N=[N2]
                  
                try:
                  
                  fed_loss_classes_1 = self.get_fed_loss_classes(
                      gt_classes_1,
                      num_fed_loss_classes=self.fed_loss_num_classes,
                      num_classes=K[0],
                      weight=self.fed_loss_cls_weights_1,
                  )
                except:
                  pass
                try:
                  fed_loss_classes_2 = self.get_fed_loss_classes(
                      gt_classes_2,
                      num_fed_loss_classes=self.fed_loss_num_classes,
                      num_classes=K[1],
                      weight=self.fed_loss_cls_weights_2,
                  )
                except:
                  pass
                  
                try:    
                  fed_loss_classes_3 = self.get_fed_loss_classes(
                      gt_classes_3,
                      num_fed_loss_classes=self.fed_loss_num_classes,
                      num_classes=K[2],
                      weight=self.fed_loss_cls_weights_3,
                  )
                except:
                  pass
                  
                  
                if not self.freeze_class2 and not self.freeze_class3:   
                  fed_loss_classes=[fed_loss_classes_1, fed_loss_classes_2, fed_loss_classes_3]
                
                if not self.freeze_class1 and not self.freeze_class2 and self.freeze_class3:
                  fed_loss_classes=[fed_loss_classes_1, fed_loss_classes_2]  
                
                if self.freeze_class2 and self.freeze_class3:
                  fed_loss_classes=[fed_loss_classes_1]  
                
                if self.freeze_class1:
                  fed_loss_classes=[fed_loss_classes_2]
                
                loss_ce=[]
                for i in range(len(fed_loss_classes)):
                  if self.freeze_class1:
                    l=i+1
                  else:
                    l=i
                  fed_loss_classes_mask = fed_loss_classes[i].new_zeros(K[l] + 1)
                  fed_loss_classes_mask[fed_loss_classes[i]] = 1
                  fed_loss_classes_mask = fed_loss_classes_mask[:K[l]]
                  weight = fed_loss_classes_mask.view(1, K[l]).expand(N[i], K[l]).float()
  
                  loss_ce.append(torch.sum(cls_loss[i] * weight) / num_boxes)
            else:
            
                loss_ce=[]
                for cls_losses in cls_loss:
                  loss = torch.sum(cls_losses) / num_boxes
                  loss_ce.append(loss)
            if not self.freeze_class2 and not self.freeze_class3:  
              losses = {'loss_ce1': loss_ce[0], 'loss_ce2': loss_ce[1], 'loss_ce3': loss_ce[2],}
            if not self.freeze_class1 and not self.freeze_class2 and self.freeze_class3:
              losses = {'loss_ce1': loss_ce[0], 'loss_ce2': loss_ce[1]}
            if self.freeze_class2 and self.freeze_class3:
              losses = {'loss_ce1': loss_ce[0]}
            if self.freeze_class1:
              losses = {"loss_ce2": loss_ce[0]}
            
              
        else:
            raise NotImplementedError

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes']
    
        batch_size = len(targets)
        pred_box_list = []
        pred_norm_box_list = []
        tgt_box_list = []
        tgt_box_xyxy_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_image_whwh = targets[batch_idx]['image_size_xyxy']
            bz_src_boxes = src_boxes[batch_idx]
            bz_target_boxes = targets[batch_idx]["boxes"]  # normalized (cx, cy, w, h)
            bz_target_boxes_xyxy = targets[batch_idx]["boxes_xyxy"]  # absolute (x1, y1, x2, y2)
            pred_box_list.append(bz_src_boxes[valid_query])
            pred_norm_box_list.append(bz_src_boxes[valid_query] / bz_image_whwh)  # normalize (x1, y1, x2, y2)
            tgt_box_list.append(bz_target_boxes[gt_multi_idx])
            tgt_box_xyxy_list.append(bz_target_boxes_xyxy[gt_multi_idx])
    
        if len(pred_box_list) != 0:
            src_boxes = torch.cat(pred_box_list)
            src_boxes_norm = torch.cat(pred_norm_box_list)  # normalized (x1, y1, x2, y2)
            target_boxes = torch.cat(tgt_box_list)
            target_boxes_abs_xyxy = torch.cat(tgt_box_xyxy_list)
            num_boxes = src_boxes.shape[0]
    
            losses = {}
            # require normalized (x1, y1, x2, y2)
            loss_bbox = F.l1_loss(src_boxes_norm, box_cxcywh_to_xyxy(target_boxes), reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes
    
            # loss_giou = giou_loss(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes_abs_xyxy))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            losses = {'loss_bbox': outputs['pred_boxes'].sum() * 0,
                      'loss_giou': outputs['pred_boxes'].sum() * 0}
    
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        
        
        
        outputs_without_aux={k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
       
        
        indices, _ = self.matcher(outputs_without_aux, targets)
        

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        try:
          num_boxes = sum(len(t["labels_1"]) for t in targets)
        except:
          num_boxes = sum(len(t["labels_2"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                
                indices, _ = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class HungarianMatcherDynamicK(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_mask: float = 1, use_focal: bool = False):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.ota_k = cfg.MODEL.DiffusionDet.OTA_K
        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.DiffusionDet.ALPHA
            self.focal_loss_gamma = cfg.MODEL.DiffusionDet.GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0,  "all costs cant be 0"
        
        self.freeze_layer1=False
        self.freeze_layer2=False
        self.freeze_layer3=False

    def forward(self, outputs, targets):
        """ simOTA for detr"""
        
        with torch.no_grad():
            try:
              bs, num_queries = outputs["pred_logits_1"].shape[:2]
            except:
              bs, num_queries = outputs["pred_logits_2"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            if self.use_focal or self.use_fed_loss:
                try:
                  out_prob_1=outputs["pred_logits_1"].sigmoid()
                except:
                  self.freeze_layer1=True
                try:
                  out_prob_2=outputs["pred_logits_2"].sigmoid()
                except:
                  self.freeze_layer2=True  
                  
                  
                try:  
                  out_prob_3=outputs["pred_logits_3"].sigmoid()  # [batch_size, num_queries, num_classes
                except:
                  self.freeze_layer3=True
                  
                out_bbox = outputs["pred_boxes"]  # [batch_size,  num_queries, 4]
            else:
                out_prob_1=outputs["pred_logits_1"].softmax(-1)  # [batch_size, num_queries, num_classes]
                try:
                  out_prob_2=outputs["pred_logits_2"].softmax(-1)
                except:
                  pass
                  
                try:
                  out_prob_3=outputs["pred_logits_3"].softmax(-1)
                except:
                  pass
                out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 4]
                
            if not self.freeze_layer2 and not self.freeze_layer3:   
              out_prob=[out_prob_1, out_prob_2, out_prob_3]
            
            if not self.freeze_layer1 and not self.freeze_layer2 and self.freeze_layer3:   
              out_prob=[out_prob_1, out_prob_2]
              
            if self.freeze_layer2 and self.freeze_layer3:   
              out_prob=[out_prob_1]    
              
            if self.freeze_layer1:
              out_prob= [out_prob_2]
            
            indices = []
            matched_ids = []
           
            assert bs == len(targets)
            for batch_idx in range(bs):
                bz_boxes = out_bbox[batch_idx]  # [num_proposals, 4]
                bz_out_prob = []
                for output_prob in out_prob:
                  bz_out_prob.append(output_prob[batch_idx])
                if not self.freeze_layer1:
                  bz_tgt_ids_1 = targets[batch_idx]["labels_1"]
                if not self.freeze_layer2:
                  bz_tgt_ids_2 = targets[batch_idx]["labels_2"]
                if not self.freeze_layer3:
                  bz_tgt_ids_3 = targets[batch_idx]["labels_3"]
                
                
                if not self.freeze_layer2 and not self.freeze_layer3:   
                  bz_tgt_ids=[bz_tgt_ids_1,bz_tgt_ids_2,bz_tgt_ids_3]
                
                if not self.freeze_layer1 and not self.freeze_layer2 and self.freeze_layer3:
                  bz_tgt_ids=[bz_tgt_ids_1,bz_tgt_ids_2]
                if self.freeze_layer2 and self.freeze_layer3: 
                  bz_tgt_ids=[bz_tgt_ids_1]      
                
                if self.freeze_layer1:
                  bz_tgt_ids=[bz_tgt_ids_2] 
                  
                try:
                  num_insts= len(bz_tgt_ids_1)
                except:
                  num_insts= len(bz_tgt_ids_2)

                if num_insts == 0:  # empty object in key frame
                    non_valid = torch.zeros(bz_out_prob[0].shape[0]).to(bz_out_prob[0]) > 0
                    indices_batchi = (non_valid, torch.arange(0, 0).to(bz_out_prob[0]))
                    matched_qidx = torch.arange(0, 0).to(bz_out_prob[0])
                    indices.append(indices_batchi)
                    matched_ids.append(matched_qidx)
                    continue 
                    
                    """
                    non_valid = []
                    indices_batchi_input=[]
                    matched_qidx=[]
                    
                    
                   
                    for bz_output_prob in bz_out_prob:
                      non_valid.append(torch.zeros(bz_output_prob.shape[0]).to(bz_output_prob) > 0)
                    
                      indices_batchi=(non_valid, torch.arange(0, 0).to(bz_output_prob))
                      indices_batchi_input.append(indices_batchi)
                      matched_qidx.append(torch.arange(0, 0).to(bz_output_prob))
                      
                    
                    indices.append(indices_batchi_input)
                    matched_ids.append(matched_qidx)
                    continue
                    """
                    
                    

                bz_gtboxs = targets[batch_idx]['boxes']  # [num_gt, 4] normalized (cx, xy, w, h)
                bz_gtboxs_abs_xyxy = targets[batch_idx]['boxes_xyxy']
                fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
                    box_xyxy_to_cxcywh(bz_boxes),  # absolute (cx, cy, w, h)
                    box_xyxy_to_cxcywh(bz_gtboxs_abs_xyxy),  # absolute (cx, cy, w, h)
                    expanded_strides=32
                )

                pair_wise_ious = ops.box_iou(bz_boxes, bz_gtboxs_abs_xyxy)

                # Compute the classification cost.
                cost_class=[]
                if self.use_focal:
                    alpha = self.focal_loss_alpha
                    gamma = self.focal_loss_gamma
                    
                    for i in range(len(bz_out_prob)):
                      neg_cost_class = (1 - alpha) * (bz_out_prob[i] ** gamma) * (-(1 - bz_out_prob[i] + 1e-8).log())
                      pos_cost_class = alpha * ((1 - bz_out_prob[i]) ** gamma) * (-(bz_out_prob[i] + 1e-8).log())
                      cost_class.append(pos_cost_class[:, bz_tgt_ids[i]] - neg_cost_class[:, bz_tgt_ids[i]])
                      #cost_class_2 = pos_cost_class[:, bz_tgt_ids_2] - neg_cost_class[:, bz_tgt_ids_2]
                      #cost_class_3 = pos_cost_class[:, bz_tgt_ids_3] - neg_cost_class[:, bz_tgt_ids_3]
                elif self.use_fed_loss:
                    # focal loss degenerates to naive one
                    for i in range(len(bz_out_prob)):
                      neg_cost_class = (-(1 - bz_out_prob[i] + 1e-8).log())
                      pos_cost_class = (-(bz_out_pro[i] + 1e-8).log())
                      #cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                      
                      cost_class.append(pos_cost_class[:, bz_tgt_ids[i]] - neg_cost_class[:, bz_tgt_ids[i]])
                    #cost_class_2 = pos_cost_class[:, bz_tgt_ids_2] - neg_cost_class[:, bz_tgt_ids_2]
                    #cost_class_3 = pos_cost_class[:, bz_tgt_ids_3] - neg_cost_class[:, bz_tgt_ids_3]
                else:
                    #cost_class = -bz_out_prob[:, bz_tgt_ids]
                    for i in range(len(bz_out_prob)):
                      cost_class.append(-bz_out_prob[i][:, bz_tgt_ids[i]])
                    #cost_class_2 = -bz_out_prob[:, bz_tgt_ids_2]
                    #cost_class_3 = -bz_out_prob[:, bz_tgt_ids_3]
                   

                # Compute the L1 cost between boxes
                # image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
                # image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
                # image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

                bz_image_size_out = targets[batch_idx]['image_size_xyxy']
                bz_image_size_tgt = targets[batch_idx]['image_size_xyxy_tgt']

                bz_out_bbox_ = bz_boxes / bz_image_size_out  # normalize (x1, y1, x2, y2)
                bz_tgt_bbox_ = bz_gtboxs_abs_xyxy / bz_image_size_tgt  # normalize (x1, y1, x2, y2)
                cost_bbox = torch.cdist(bz_out_bbox_, bz_tgt_bbox_, p=1)

                cost_giou = -generalized_box_iou(bz_boxes, bz_gtboxs_abs_xyxy)

                # Final cost matrix
               
               
               
                length = len(bz_tgt_ids)
                self.cost_class=1/length
                if not self.freeze_layer2 and not self.freeze_layer3:   
                  cost = self.cost_bbox * cost_bbox +  self.cost_class * cost_class[0] +  self.cost_class * cost_class[1] +  self.cost_class * cost_class[2] + self.cost_giou * cost_giou + 100.0 * (~is_in_boxes_and_center)  
                  
                  
                if not self. freeze_layer1 and not self.freeze_layer2 and self.freeze_layer3:   
                  cost = self.cost_bbox * cost_bbox +  self.cost_class * cost_class[0] +  self.cost_class * cost_class[1] + self.cost_giou * cost_giou + 100.0 * (~is_in_boxes_and_center)
                  
                    
                  
                  
                if self.freeze_layer2 and self.freeze_layer3:   
                  cost = self.cost_bbox * cost_bbox +  self.cost_class * cost_class[0] + self.cost_giou * cost_giou + 100.0 * (~is_in_boxes_and_center)   
                  
                  
                if self.freeze_layer1:   
                  cost = self.cost_bbox * cost_bbox +  self.cost_class * cost_class[0] + self.cost_giou * cost_giou + 100.0 * (~is_in_boxes_and_center)      
                # cost = (cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center))  # [num_query,num_gt]
                cost[~fg_mask] = cost[~fg_mask] + 10000.0
                
                

                # if bz_gtboxs.shape[0]>0:
                indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0])

                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)

        return indices, matched_ids

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        xy_target_gts = box_cxcywh_to_xyxy(target_gts)  # (x1, y1, x2, y2)

        anchor_center_x = boxes[:, 0].unsqueeze(1)
        anchor_center_y = boxes[:, 1].unsqueeze(1)

        # whether the center of each anchor is inside a gt box
        b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0)
        b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
        b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] ,
        is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_boxes_all = is_in_boxes.sum(1) > 0  # [num_query]
        # in fixed center
        center_radius = 2.5
        # Modified to self-adapted sampling --- the center size depends on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212
        b_l = anchor_center_x > (target_gts[:, 0] - (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = anchor_center_x < (target_gts[:, 0] + (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = anchor_center_y > (target_gts[:, 1] - (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = anchor_center_y < (target_gts[:, 1] + (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost)  # [300,num_gt]
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = self.ota_k

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)

        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1

        while (matching_matrix.sum(0) == 0).any():
            num_zero_gt = (matching_matrix.sum(0) == 0).sum()
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0:  # If a query matches more than one gt
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                                           dim=1)  # find gt for these queries with minimal cost
                matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1  # keep gt with minimal cost

        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        matched_query_id = torch.min(cost, dim=0)[1]

        return (selected_query, gt_indices), matched_query_id
