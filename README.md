# dcase2022_task2_evaluator
The **dcase2022_task2_evaluator** is a script for calculating the AUC, pAUC, precision, recall, and F1 scores from the anomaly score list for the [evaluation dataset](https://zenodo.org/record/6586456) in DCASE 2022 Challenge Task 2 "Unsupervised Anomalous Sound Detection for Machine Condition Monitoring Applying Domain Generalization Techniques".

https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring

## Description

The **dcase2022_task2_evaluator** consists of only one script:
- `dcase2022_task2_evaluator.py`
    - This script outputs the AUC and pAUC scores by using:
      - Ground truth of the normal and anomaly labels
      - Anomaly scores for each wave file listed in the csv file for each macine type, section, and domain
      - Detection results for each wave file listed in the csv file for each macine type, section, and domain

## Usage
### 1. Clone repository
Clone this repository from Github.

### 2. Prepare data
- Anomaly scores
    - Generate csv files `anomaly_score_<machine_type>_section_<section_index>.csv` and `decision_result_<machine_type>_section_<section_index>.csv` by using a system for the [evaluation dataset](https://zenodo.org/record/6586456). (The format information is described [here](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring#submission).)
- Rename the directory containing the csv files to a team name
- Move the directory into `./teams/`

### 3. Check directory structure
- ./dcase2022_task2_evaluator
    - /dcase2022_task2_evaluator.py
    - /ground_truth_data
        - ground_truth_bearing_section_03_test.csv
        - ground_truth_bearing_section_04_test.csv
        - ...
    - /teams
        - /<team_name_1>
            - anomaly_score_bearing_section_03_test.csv
            - anomaly_score_bearing_section_04_test.csv
            - ...
            - decision_result_valve_section_04_test.csv
            - decision_result_valve_section_05_test.csv
        - /<team_name_2>
            - anomaly_score_bearing_section_03_test.csv
            - anomaly_score_bearing_section_04_test.csv
            - ...
            - decision_result_valve_section_04_test.csv
            - decision_result_valve_section_05_test.csv
        - ...
    - /teams_result
        - *<team_name_1>_result.csv*
        - *<team_name_2>_result.csv*
        - ...
    - /README.md


### 4. Change parameters
The parameters are defined in the script `dcase2022_task2_evaluator.py` as follows.
- **MAX_FPR**
    - The FPR threshold for pAUC : default 0.1
- **RESULT_DIR**
    - The output directory : default `./teams_result/`

### 5. Run script
Run the script `dcase2022_task2_evaluator.py`
```
$ python dcase2022_task2_evaluator.py
```
The script `dcase2022_task2_evaluator.py` calculates the AUC, pAUC, precision, recall, and F1 scores for each machine type, section, and domain and output the calculated scores into the csv files (`<team_name_1>_result.csv`, `<team_name_2>_result.csv`, ...) in **RESULT_DIR** (default: `./teams_result/`).

### 6. Check results
You can check the AUC, pAUC, precision, recall, and F1 scores in the `<team_name_N>_result.csv` in **RESULT_DIR**.
The AUC, pAUC, precision, recall, and F1 scores for each machine type, section, and domain are listed as follows:

`result_<team_name_N>.csv`
```
ToyCar
section,AUC (all),AUC (source),AUC (target),pAUC,precision (source),precision (target),recall (source),recall (target),F1 score (source),F1 score (target)
03,0.7579,0.8204,0.6954,0.71,0.6666666666666666,0.7575757575757576,0.28,1.0,0.3943661971830986,0.8620689655172413
04,0.5183,0.6164,0.4202,0.4942105263157895,0.6333333333333333,0.46296296296296297,0.38,0.5,0.4750000000000001,0.4807692307692307
05,0.6746,0.8371999999999999,0.512,0.6447368421052632,0.8333333333333334,0.5476190476190477,0.3,0.92,0.4411764705882353,0.6865671641791046
,,AUC,pAUC,precision,recall,F1 score
arithmetic mean,,0.6502666666666667,0.6163157894736843,0.6502485169151835,0.5633333333333334,0.5566580047061518
harmonic mean,,0.6118288605395442,0.6020590065980902,0.6259758490782256,0.44042200910247076,0.5170556917115303
source harmonic mean,,0.8204,0.71,0.6666666666666666,0.28,0.3943661971830986
target harmonic mean,,0.5822249313921185,0.5842930428015208,0.6184265643220234,0.4974199423281226,0.5513620450947704

...

bearing
section,AUC (all),AUC (source),AUC (target),pAUC,precision (source),precision (target),recall (source),recall (target),F1 score (source),F1 score (target)
03,0.5661,0.6338,0.49839999999999995,0.47368421052631576,0.4888888888888889,0.525,0.88,0.84,0.6285714285714286,0.6461538461538462
04,0.7188000000000001,0.772,0.6656,0.66,0.6481481481481481,0.6031746031746031,0.7,0.76,0.673076923076923,0.672566371681416
05,0.5995,0.7701999999999999,0.4288,0.5168421052631579,0.5833333333333334,0.5,0.28,0.94,0.3783783783783784,0.6527777777777778
,,AUC,pAUC,precision,recall,F1 score
arithmetic mean,,0.6281333333333333,0.5501754385964913,0.5580908289241623,0.7333333333333334,0.608587454273295
harmonic mean,,0.5992569463670572,0.5394626348869298,0.5522190254909032,0.6181450872818578,0.5833252643095251
source harmonic mean,,0.6338,0.47368421052631576,0.4888888888888889,0.8799999999999999,0.6285714285714286
target harmonic mean,,0.5927952992183491,0.5548731904323784,0.5669063112727686,0.5834240736707836,0.5750466019875952

...

,,AUC,pAUC,precision,recall,F1 score
"arithmetic mean over all machine types, sections, and domains",,0.5793380952380952,0.5339348370927318,0.46742026804033576,0.4823809523809524,0.43567479273374615
"harmonic mean over all machine types, sections, and domains",,0.5301047626828244,0.5280441052754629,1.332267629550185e-15,1.3322676295501841e-15,1.3322676295501845e-15
"source harmonic mean over all machine types, sections, and domains",,0.6423240401028497,0.5280441052754629,1.1657341758564126e-15,1.1657341758564108e-15,1.1657341758564118e-15
"target harmonic mean over all machine types, sections, and domains",,0.45126505654047017,0.5280441052754629,1.554312234475215e-15,1.5543122344752146e-15,1.554312234475215e-15

official score,,0.5294160921841315
official score ci95,,2.9525671289513583e-06
```

## Citation
If you use this system, please cite all the following three papers:
- Kota Dohi, Keisuke Imoto, Noboru Harada, Daisuke Niizumi, Yuma Koizumi, Tomoya Nishida, Harsh Purohit, Takashi Endo, Masaaki Yamamoto, and Yohei Kawaguchi. Description and discussion on DCASE 2022 challenge task 2: unsupervised anomalous sound detection for machine condition monitoring applying domain generalization techniques. In arXiv e-prints: 2206.05876, 2022. [URL](https://arxiv.org/abs/2206.05876)
- Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, Shoichiro Saito, "ToyADMOS2: Another Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection under Domain Shift Conditions," in arXiv e-prints: 2106.02369, 2021. [URL](https://arxiv.org/abs/2106.02369)
- Kota Dohi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, Yuki Nikaido, and Yohei Kawaguchi. MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task. In arXiv e-prints: 2205.13879, 2022. [URL](https://arxiv.org/abs/2205.13879)
