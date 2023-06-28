from evalutils import DetectionEvaluation
from evalutils.io import CSVLoader
from evalutils.validators import ExpectedColumnNamesValidator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from typing import Dict
import os


class DentexChallenge:
    def __init__(self, categories, prediction_file, gt_file, output_file='/output/metrics.json'):
        self.categories = categories
        self.prediction_file = prediction_file
        self.gt_file = gt_file
        self.output_file = output_file
        self._case_results = {}
        self._aggregate_results = {}

    def load_data(self):
        with open(self.gt_file) as f:
            self.gt_data = json.load(f)
        with open(self.prediction_file) as f:
            self.prediction_data = json.load(f)

    def separate_data_gt(self, dat):
        separated_data = {}
        for cat in self.categories:
            cat_list = cat.split("_")
            
            cat_id_name = cat_list[0][:-3] + "y_id_" + cat_list[1]
            category_name = cat_list[1]
            data=dat.copy()
            separated_data[category_name] = {
                'images': data['images'],
                'annotations': [anno for anno in data['annotations']],
                'categories': data[cat]
            }
            for anno in separated_data[category_name]['annotations']:
                anno['category_id'] = anno[cat_id_name]

            separated_data[category_name]['file_path'] = f"/tmp/separated_data_{category_name}.json"
            with open(separated_data[category_name]['file_path'], "w") as f:
                f.write(json.dumps(separated_data[category_name]))
        return separated_data

    def separate_data_predict(self, data):
        separated_data_predict = {}

        for i in range(len(self.categories)):
            cat = self.categories[i]
            cat_list = cat.split("_")


            cat_id_name = cat_list[0][:-3] + "y_id_" + cat_list[1]
            category_name = cat_list[1]

            output_anno=[]
            data_boxes = data["boxes"]
            for anno in data_boxes:
                bbox = [anno["corners"][0][0],anno["corners"][0][1],anno["corners"][3][0],anno["corners"][3][1]]

                bbox[2] = bbox[2]-bbox[0]
                bbox[3] = bbox[3]-bbox[1]

                anno_ind={
                "image_id": anno["corners"][0][2],
                "category_id": int(anno["name"].split("-")[i]),
                "bbox": bbox,
                "score": anno["probability"]
                }
                output_anno.append(anno_ind)
            separated_data_predict[category_name] = {
                'annotations': output_anno
            }
            separated_data_predict[category_name]['file_path'] = f"/tmp/separated_data_predict_{category_name}.json"
            with open(separated_data_predict[category_name]['file_path'], "w") as f:
                f.write(json.dumps(separated_data_predict[category_name]["annotations"]))
        return separated_data_predict

    def score(self):
        metrics_names = ['AP50', 'AP75', 'AP', 'AR']

        separated_gt = self.separate_data_gt(self.gt_data)

        job_pk = self.prediction_data[0]["pk"]
        r_path = self.prediction_data[0]["outputs"][0]["interface"]["relative_path"]

        print(r_path)
        prediction_path = f"/input/{job_pk}/output/{r_path}"
        with open(prediction_path) as f:
            self.prediction_data = json.load(f)
       
        separated_prediction = self.separate_data_predict(self.prediction_data)

        for cat in self.categories:
            category_name = cat.split('_')[1]

            gt = COCO(separated_gt[category_name]['file_path'])

            prediction = gt.loadRes(separated_prediction[category_name]['file_path'])

            cocoEval = COCOeval(gt, prediction, 'bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            dict_stats = {
            'AP' : cocoEval.stats[0],
            'AP50' : cocoEval.stats[1],
            'AP75' : cocoEval.stats[2],
            'AR': cocoEval.stats[8]

            }
            self._case_results[category_name] = dict_stats 

        # calculate aggregate metrics
        for metric in metrics_names:
            self._aggregate_results[metric] = sum([self._case_results[cat][metric] for cat in self._case_results]) / len(self._case_results)

    def save(self):
        metrics = {}
        for category in self._case_results:
            if category =="1":
                cat_name = "Quadrant"
            if category =="2":
                cat_name = "Enumeration"
            if category =="3":
                cat_name = "Diagnosis"
            metrics[f"{cat_name}"] = self._case_results[category]
        metrics["Aggregates"] = self._aggregate_results
        with open(self.output_file, "w") as f:
            f.write(json.dumps(metrics))

    def evaluate(self):
        self.load_data()
        self.score()
        self.save()


if __name__ == "__main__":
    categories = ['categories_1', 'categories_2', 'categories_3']

    input_folder = '/input/'
    gt_folder = '/opt/app/ground-truth/'

    # Get a list of all JSON files in the input folder
    input_files = "predictions.json"

    # Get a list of all JSON files in the ground-truth folder
    gt_files = [file for file in os.listdir(gt_folder) if file.endswith('.json')]

    # Select the first JSON file from the input folder
    if input_files:
        prediction_file = input_folder + input_files
    else:
        raise FileNotFoundError("No JSON files found in the input folder.")

    # Select the first JSON file from the ground-truth folder
    if gt_files:
        gt_file = gt_folder + gt_files[0]
    else:
        raise FileNotFoundError("No JSON files found in the ground-truth folder.")

    DentexChallenge(categories, prediction_file, gt_file).evaluate()
