import os
import json
from detectron2.config import get_cfg
from hierarchialdet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from hierarchialdet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from hierarchialdet.predictor import VisualizationDemo
import argparse
import SimpleITK as sitk
import glob


list_ids = [
                          {"height": 1316, "width": 2892, "id": 1, "file_name": "val_15.png"},
                          {"height": 1316, "width": 2942, "id": 2, "file_name": "val_38.png"},
                          {"height": 1316, "width": 2987, "id": 3, "file_name": "val_33.png"},
                          {"height": 1504, "width": 2872, "id": 4, "file_name": "val_30.png"},
                          {"height": 1316, "width": 2970, "id": 5, "file_name": "val_5.png"},
                          {"height": 1316, "width": 2860, "id": 6, "file_name": "val_21.png"},
                          {"height": 1504, "width": 2804, "id": 7, "file_name": "val_39.png"},
                          {"height": 1316, "width": 2883, "id": 8, "file_name": "val_46.png"},
                          {"height": 1316, "width": 2967, "id": 9, "file_name": "val_20.png"},
                          {"height": 1504, "width": 2872, "id": 10, "file_name": "val_3.png"},
                          {"height": 1316, "width": 2954, "id": 11, "file_name": "val_29.png"},
                          {"height": 976, "width": 1976, "id": 12, "file_name": "val_2.png"},
                          {"height": 1316, "width": 2870, "id": 13, "file_name": "val_16.png"},
                          {"height": 1316, "width": 3004, "id": 14, "file_name": "val_25.png"},
                          {"height": 1316, "width": 2745, "id": 15, "file_name": "val_24.png"},
                          {"height": 1504, "width": 2872, "id": 16, "file_name": "val_31.png"},
                          {"height": 1316, "width": 2782, "id": 17, "file_name": "val_26.png"},
                          {"height": 1316, "width": 2744, "id": 18, "file_name": "val_44.png"},
                          {"height": 1504, "width": 2872, "id": 19, "file_name": "val_27.png"},
                          {"height": 1504, "width": 2868, "id": 20, "file_name": "val_41.png"},
                          {"height": 1316, "width": 3000, "id": 21, "file_name": "val_37.png"},
                          {"height": 1316, "width": 2797, "id": 22, "file_name": "val_40.png"},
                          {"height": 1316, "width": 2930, "id": 23, "file_name": "val_6.png"},
                          {"height": 1316, "width": 3003, "id": 24, "file_name": "val_18.png"},
                          {"height": 1316, "width": 2967, "id": 25, "file_name": "val_13.png"},
                          {"height": 1316, "width": 2822, "id": 26, "file_name": "val_8.png"},
                          {"height": 1316, "width": 2836, "id": 27, "file_name": "val_49.png"},
                          {"height": 1316, "width": 2704, "id": 28, "file_name": "val_23.png"},
                          {"height": 976, "width": 1976, "id": 29, "file_name": "val_1.png"},
                          {"height": 1504, "width": 2872, "id": 30, "file_name": "val_43.png"},
                          {"height": 1504, "width": 2872, "id": 31, "file_name": "val_28.png"},
                          {"height": 1504, "width": 2872, "id": 32, "file_name": "val_19.png"},
                          {"height": 1316, "width": 2728, "id": 33, "file_name": "val_14.png"},
                          {"height": 1316, "width": 2747, "id": 34, "file_name": "val_32.png"},
                          {"height": 976, "width": 1976, "id": 35, "file_name": "val_36.png"},
                          {"height": 1316, "width": 2829, "id": 36, "file_name": "val_47.png"},
                          {"height": 1316, "width": 2846, "id": 37, "file_name": "val_48.png"},
                          {"height": 1536, "width": 3076, "id": 38, "file_name": "val_17.png"},
                          {"height": 976, "width": 1976, "id": 39, "file_name": "val_42.png"},
                          {"height": 1504, "width": 2884, "id": 40, "file_name": "val_45.png"},
                          {"height": 1316, "width": 2741, "id": 41, "file_name": "val_9.png"},
                          {"height": 1316, "width": 2794, "id": 42, "file_name": "val_4.png"},
                          {"height": 1316, "width": 2959, "id": 43, "file_name": "val_34.png"},
                          {"height": 1316, "width": 2874, "id": 44, "file_name": "val_10.png"},
                          {"height": 1316, "width": 2978, "id": 45, "file_name": "val_35.png"},
                          {"height": 1504, "width": 2884, "id": 46, "file_name": "val_11.png"},
                          {"height": 1316, "width": 2794, "id": 47, "file_name": "val_12.png"},
                          {"height": 1316, "width": 2959, "id": 48, "file_name": "val_7.png"},
                          {"height": 1316, "width": 2912, "id": 49, "file_name": "val_22.png"},
                          {"height": 1504, "width": 2872, "id": 50, "file_name": "val_0.png"},
                      ]

def custom_format_output(outputs, img_ids):
    boxes = []
    for k, instances in enumerate(outputs):
        for i in range(len(instances)):
            instance = instances[i]
            bbox_coords = instance.pred_boxes.tensor[0].tolist()

            category_id_1 = instance.pred_classes_1[0].item()
            category_id_2 = instance.pred_classes_2[0].item()
            category_id_3 = instance.pred_classes_3[0].item()
            img_id = img_ids[k]
            box = {
                "name": f"{category_id_1} - {category_id_2} - {category_id_3}",
                "corners": [
                    [bbox_coords[0], bbox_coords[1], img_id],
                    [bbox_coords[0], bbox_coords[3], img_id],
                    [bbox_coords[2], bbox_coords[1], img_id],
                    [bbox_coords[2], bbox_coords[3], img_id]
                ],
                "probability": instance.scores[0].item(),
            }
            boxes.append(box)

        custom_annotations={
        "name": "Regions of interest",
        "type": "Multiple 2D bounding boxes",
        "boxes": boxes,
        "version": { "major": 1, "minor": 0 }
        }
    return custom_annotations


def coco_format_output(outputs,img_ids):
    coco_annotations = []
    for k, instances in enumerate(outputs):
        for i in range(len(instances)):
            instance = instances[i]
            bbox_coords = instance.pred_boxes.tensor[0].tolist()
            bbox_coords[2] = bbox_coords[2] - bbox_coords[0]
            bbox_coords[3] = bbox_coords[3] - bbox_coords[1]

            coco_annotation = {
                            "image_id": img_ids[k],
                            "category_id_1": instance.pred_classes_1[0].item(),
                            "category_id_2": instance.pred_classes_2[0].item(),
                            "category_id_3": instance.pred_classes_3[0].item(),
                            "bbox": bbox_coords,
                            "score": instance.scores[0].item(),
                        }
            coco_annotations.append(coco_annotation)
    return coco_annotations


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")


    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "--nclass",
        type=int,
        default=3,
        help="Number of trained classes",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class Hierarchialdet:
    def __init__(self):
        self.cfg = None
        self.demo = None
        self.input_dir = "input"

    def setup(self):
        args = get_parser().parse_args()
        self.cfg = get_cfg()
        add_diffusiondet_config(self.cfg)
        add_model_ema_configs(self.cfg)
        self.cfg.merge_from_file("/opt/app/configs/diffdet.custom.swinbase.nonpretrain.yaml")
        self.cfg.MODEL.WEIGHTS = "/opt/app/pretrained_model/model_final.pth"
        self.cfg.merge_from_list(args.opts)
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
        self.cfg.freeze()
        self.demo = VisualizationDemo(self.cfg, k=2)

    def process(self):
        self.setup()

        #image_files = [f for f in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, f))]

        all_outputs = []
        img_ids = []
        
        file_path = glob.glob('/input/images/panoramic-dental-xrays/*.mha')[0]
        image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image)
        print("test..")
        for k in range(image_array.shape[2]):
            image_name = "val_{}.png".format(k)
            predictions, _ = self.demo.run_on_image(image_array[:,:,k,:])
            instances = predictions["instances"]
            all_outputs.append(instances)
            for input_img in list_ids:
                if input_img["file_name"] == image_name:
                    img_id = input_img["id"]
            img_ids.append(img_id)
        coco_annotations = custom_format_output(all_outputs,img_ids)

        output_file = "/output/abnormal-teeth-detection.json"
        with open(output_file, "w") as f:
            json.dump(coco_annotations, f)

        print("Inference completed. Results saved to", output_file)


if __name__ == "__main__":
    Hierarchialdet().process()
