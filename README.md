# DENTEX CHALLENGE 2023 
Dental Enumeration and Diagnosis on Panoramic X-rays Challenge


<details><summary>Table of Contents</summary><p>
  
* [Our Framework](#Our-Framework)
* [Output of the Model](#Output-of-the-Model)
* [What is Our Approach?](#what-is-our-approach)
* [Citing Us](#citing-us)
* [Data](#data)
* [License](#license)
* [Contact](#contact)
  
</p></details><p></p>

## Our Framework
![Our method relies on a hierarchical learning approach utilizing a combi- nation of multi-label detection, bounding box manipulation, and weight transfer.](figures/data.png)
*Our Diffusion-Based Object Detection method relies on a hierarchical learning approach utilizing a combination of multi-label detection, bounding box manipulation and weight transfer.*
## Output of the Model
![Output from our final model showing well-defined boxes for diseased teeth with corresponding quadrant (Q), enumeration (N), and diagnosis (D) labels., etc.](figures/output.png)
*Output from our final model showing well-defined boxes for diseased teeth with corresponding quadrant (Q), enumeration (N), and diagnosis (D) labels.*

## What is Our Approach?
We present the Dental Enumeration and Diagnosis on Panoramic X-rays Challenge (DENTEX), organized in conjunction with the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) in 2023. The primary objective of this challenge is to develop algorithms that can accurately detect abnormal teeth with dental enumeration and associated diagnosis. This not only aids in accurate treatment planning but also helps practitioners carry out procedures with a low margin of error.

The challenge provides three types of hierarchically annotated data and additional unlabeled X-rays for optional pre-training. The annotation of the data is structured using the Fédération Dentaire Internationale (FDI) system. The first set of data is partially labeled because it only includes quadrant information. The second set of data is also partially labeled but contains additional enumeration information along with the quadrant. The third data is fully labeled because it includes all quadrant-enumeration-diagnosis information for each abnormal tooth, and all participant algorithms will be benchmarked on the third data. 

DENTEX aims to provide insights into the effectiveness of AI in dental radiology analysis and its potential to improve dental practice by comparing frameworks that simultaneously point out abnormal teeth with dental enumeration and associated diagnosis on panoramic dental X-rays.

*Please visit our website to join[DENTEX (Dental Enumeration and Diagnosis on Panoramic X- rays Challenge)](https://dentex.grand-challenge.org/) which will be held at MICCAI 2023.*

## Citing Us

If you use HierarchicalDet, we would appreciate references to the following papers. 

1. **Ibrahim Ethem Hamamci, Sezgin Er, Enis Simsar, Anjany Sekuboyina, Mustafa Gundogar, Bernd Stadlinger, Albert Mehl, Bjoern Menze., Diffusion-Based Hierarchical Multi-Label Object Detection to Analyze Panoramic Dental X-rays, 2023.**<br  />Pre-print: https://arxiv.org/abs/2303.06500


## Data

* The dataset will be released later as part of [DENTEX (Dental Enumeration and Diagnosis on Panoramic X- rays Challenge)](https://dentex.grand-challenge.org/). 

**Note**: The data used and annotations are fully identical to the one that will be used for the MICCAI challenge. Therefore, our work also serves as a baseline method for [DENTEX (Dental Enumeration and Diagnosis on Panoramic X- rays Challenge)](https://dentex.grand-challenge.org/) which will be held at MICCAI 2023.

## License
The data is provided under the [CC BY-SA 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/), making it fully open-sourced.

The rest of this repository is under the [MIT License](https://choosealicense.com/licenses/mit/).


## Contact
For queries and issues not fit for a github issue, please email [Ibrahim Ethem Hamamci](mailto:ibrahim.hamamci@uzh.ch).
