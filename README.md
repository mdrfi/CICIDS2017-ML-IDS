# CICIDS2017-ML-IDS

گزارش فارسی پروژه در فایل [`REPORT_FA.md`](REPORT_FA.md) قرار دارد.

A network intrusion detection system (IDS) that uses machine learning. The
system reads network-flow records from the CIC-IDS2017 dataset. It tells you if
a flow is normal traffic or an attack. If the flow is an attack, the system
also tells you which attack it is.

An IDS is a program that looks at network traffic and finds attacks. A flow is
one network conversation between two computers. It is not one packet.

---

## 1. Project overview

The project is the final project for the Network Security course. It follows
the two documents in the parent folder:

- `cns- FinalProject-S2026.pdf` - the project description.
- `New Intrusion Detection Dataset.pdf` - the paper of Sharafaldin, Lashkari,
  and Ghorbani (ICISSP 2018) that introduces the CIC-IDS2017 dataset.

The complete implementation is in the `notebooks/` folder. Each notebook is one
stage of the work. You can read every step in the notebook. The notebooks do
not hide any calculation in an imported library.

## 2. Project objectives

1. Read the CIC-IDS2017 CSV files and prepare them for machine learning.
2. Find the 20 most useful features with three methods.
3. Train a Random Forest model to separate normal traffic from attacks.
4. Train a Decision Tree model and a K-Nearest Neighbors model for comparison.
5. Train a second Random Forest model to name the attack.
6. Measure all models with standard metrics.
7. Explain why the models make their decisions.

## 3. Dataset

The dataset is CIC-IDS2017 from the Canadian Institute for Cybersecurity.
Download it from <https://www.unb.ca/cic/datasets/ids-2017.html>.

The project uses the eight files in `MachineLearningCSV`. The authors made
these files with the CICFlowMeter program. Each row is one network flow. Each
row has 78 measured values and one label. The paper reports more than 80
features, because it also counts the flow identifiers. These CSV files do not
contain the identifiers.

Two of the 78 columns are equal: the files contain `Fwd Header Length` twice.
The pipeline removes the copy, so 77 features remain.

The capture ran for five days, from Monday to Friday. Monday has normal traffic
only. The attacks run on the other four days.

### Data after cleaning

| Item | Value |
| --- | --- |
| Rows in the CSV files | 2,830,743 |
| Invalid rows removed | 2,867 |
| Duplicate rows removed | 329,896 |
| Rows kept | 2,497,980 |
| Training rows (80%) | 1,998,384 |
| Test rows (20%) | 499,596 |
| Numeric features | 77 |

## 4. Supported attack classes

The first stage uses two classes: **benign** (0) and **attack** (1).

The second stage uses 12 attack classes. Table 3 of the CIC-IDS2017 paper
reports one combined `Web Attack` class, so this project does the same. The
three web-attack labels (Brute Force, XSS, and SQL Injection) become one class.

| Class | Rows after cleaning |
| --- | --- |
| BENIGN | 2,072,254 |
| DoS Hulk | 172,846 |
| DDoS | 128,014 |
| PortScan | 90,694 |
| DoS GoldenEye | 10,282 |
| FTP-Patator | 5,931 |
| DoS slowloris | 5,374 |
| DoS Slowhttptest | 5,228 |
| SSH-Patator | 3,219 |
| Web Attack | 2,143 |
| Bot | 1,948 |
| Infiltration | 36 |
| Heartbleed | 11 |

The processed data keeps three label columns:

| Column | Content | Used by |
| --- | --- | --- |
| `is_attack` | 0 or 1 | binary intrusion detection |
| `attack_category` | `BENIGN` or one of the 12 attack classes | attack-type classification |
| `attack_label` | the original CIC-IDS2017 label | reference |

## 5. System architecture

The system first performs binary intrusion detection. The attack-type
classifier receives only the flows identified as attacks.

```text
CIC-IDS2017 CSV files
        |
   Data preprocessing        (cleaning, split, scaling)
        |
   Feature selection         (77 features -> 20 features)
        |
   Binary Intrusion Detection: Random Forest    benign or attack
        |
        +-- benign ------------------------> report "BENIGN"
        |
        +-- attack --> Attack-Type Classification: Random Forest --> report the attack class
        |
   Performance evaluation
```

Binary intrusion detection and attack-type classification use the same 20 input features.

## 6. Data preprocessing

Notebook `00_complete_pipeline.ipynb` does these steps in order:

1. **Read** the eight CSV files and join them into one table.
2. **Clean the column names.** The original names contain extra spaces.
3. **Repair the labels.** The published files contain a damaged character in
   the three web-attack labels. The notebook changes this character to a
   hyphen. It does not change any other label, so `FTP-Patator` keeps its
   original name.
4. **Convert every feature to a number.** A value that is not a number
   becomes empty.
5. **Remove invalid rows.** A row is invalid when a value is missing or
   infinite. A flow with a duration of zero gives an infinite rate.
6. **Remove one repeated column.** The files contain `Fwd Header Length`
   twice. The notebook first makes sure that the two columns are equal.
7. **Change the numbers to `float32`.** This halves the memory use.
8. **Remove duplicate rows.** This step runs **before** the split. A duplicate
   flow that stays in the data can go into the training set and the test set at
   the same time. The model then repeats an answer that it already knows, and
   the test score becomes too high.
9. **Make the labels.** `is_attack` becomes 0 for `BENIGN` and 1 for every
   attack. `attack_category` groups the three web attacks into one class.
10. **Split the data** into 80% training data and 20% test data.
11. **Scale the features** with `StandardScaler`.
12. **Save** `train.parquet`, `test.parquet`, the scaler, and a JSON report.

### Two rules that protect the results

**The split is stratified by attack label.** A split that uses only the binary
label can put all records of a small class into one partition. Heartbleed has
11 records, SQL Injection has 21, and Infiltration has 36. Stratification by
attack label keeps every class in both partitions. It also keeps the same
benign/attack ratio in both partitions.

**The scaler learns from the training data only.** The notebook calls `fit` on
the training rows and `transform` on the test rows. The test data does not
influence any part of the pipeline.

Notebook 00 prints two checks after the split:

1. The scaled training data must have mean 0 and standard deviation 1. The
   check ignores the constant columns, because a constant column has no
   variance and stays at zero.
2. No flow may be in the training set and in the test set at the same time. The
   check runs on the cleaned data, before scaling. The cleaned data must have
   no repeated row, and the two partitions together must have exactly the rows
   of the cleaned data.

The notebook then reports one more number. The saved files hold scaled values
in `float32` format. Scaling and rounding give 99 test rows the same saved
value as a training row. These flows are different in the original data, so
they are not a data leak. They are 0.02% of the test data, and they show the
limit of the `float32` format.

## 7. Feature selection

Notebook 00 also selects the features. It uses a random sample of 200,000
**training** rows. It never reads the test data.

### Step 1: remove the constant features

A constant feature has the same value in every row. It cannot separate the two
classes. Eight features are constant, so 69 features remain.

### Step 2: score every feature with three methods

| Method | What it measures |
| --- | --- |
| Correlation | the strength of a straight-line relation with the label |
| Mutual information | any relation with the label, also one that is not a straight line |
| Random Forest importance | how much the feature helps the trees separate the classes |

Each score is scaled to the range 0 to 1. The combined score is the mean of the
three scaled values.

### Step 3: remove the features that repeat each other

The project description asks for a correlation analysis **between the
features**. CIC-IDS2017 contains groups of features that measure the same
quantity. Two examples:

- `Subflow Bwd Bytes` and `Total Length of Bwd Packets` are equal.
- `Avg Bwd Segment Size` and `Bwd Packet Length Mean` are equal.

The notebook reads the features in rank order. It keeps a feature only when the
absolute correlation with every kept feature is below 0.95. This step removes
22 features. Without this step, the 20 selected features contain several equal
pairs, so the models receive fewer than 20 different signals.

### Selected features

The 20 features below are the model input. The list is in
`artifacts/feature_selection/selected_features.json`.

| # | Feature | # | Feature |
| --- | --- | --- | --- |
| 1 | Packet Length Variance | 11 | Fwd Packet Length Max |
| 2 | Packet Length Std | 12 | Idle Min |
| 3 | Avg Bwd Segment Size | 13 | Init_Win_bytes_backward |
| 4 | Bwd Packet Length Std | 14 | Fwd IAT Total |
| 5 | Average Packet Size | 15 | Avg Fwd Segment Size |
| 6 | Destination Port | 16 | Total Length of Fwd Packets |
| 7 | Fwd IAT Max | 17 | Init_Win_bytes_forward |
| 8 | Fwd IAT Std | 18 | Fwd IAT Mean |
| 9 | Subflow Bwd Bytes | 19 | Flow IAT Mean |
| 10 | Flow IAT Std | 20 | Min Packet Length |

`IAT` means inter-arrival time. It is the time between two packets.

The project description names flow duration, packet counts, transferred bytes,
average packet length, and the destination port as important. The list above
covers four of these five properties:

| Property | Feature in the list |
| --- | --- |
| Flow duration | `Fwd IAT Total` |
| Transferred bytes | `Total Length of Fwd Packets`, `Subflow Bwd Bytes` |
| Average packet length | `Average Packet Size` |
| Destination port | `Destination Port` |
| Packet counts | none |

`Flow Duration` itself is not in the list. Its correlation with
`Fwd IAT Total` is 0.999, so the two features give the same information, and
the pipeline keeps only the better ranked one.

The list contains no packet count. `Total Fwd Packets` has rank 44 and
`Total Backward Packets` has rank 52, so both stay outside the first 20. Both
also repeat `Subflow Bwd Bytes`, which is in the list.

The list agrees with Table 3 of the paper, which names packet-length
statistics, initial window bytes, and inter-arrival times as the best features
for most attack families.

## 8. Models

| Model | Role | Main settings |
| --- | --- | --- |
| Random Forest | main binary intrusion-detection model | 100 trees, maximum depth 20, balanced class weights |
| Decision Tree | comparison model | maximum depth 20, balanced class weights |
| K-Nearest Neighbors | comparison model | 5 neighbors, distance weights, own scaler |
| Random Forest | attack-type classification model | 150 trees, maximum depth 25, balanced class weights |

The attack class is much smaller than the benign class. Balanced class weights
give the attack class more influence. Without them, a model can call almost
everything benign and still get a high accuracy.

Every model uses `random_state = 42`, so a new run gives the same result.

## 9. Training procedure

**Binary Intrusion Detection** (notebook `03_train_models.ipynb`):

- Random Forest and Decision Tree train on all 1,998,384 training rows.
- K-Nearest Neighbors trains on a stratified sample of 25,000 rows. This model
  keeps all training rows in memory and compares each test row with all of
  them. A smaller sample keeps the time and the memory reasonable.
- All three models are measured on the complete test set of 499,596 rows.
- The prediction runs in batches of 20,000 rows.

**Attack-Type Classification** (notebook `04_attack_type_classification.ipynb`):

- The model trains on the 340,581 attack rows of the training data.
- The model is measured on the 85,145 attack rows of the test data.
- Benign rows are removed, because benign traffic has no attack type.

## 10. Evaluation metrics

| Metric | Meaning |
| --- | --- |
| Accuracy | the part of all flows with a correct answer |
| Precision | of all attack alerts, how many are correct |
| Recall | of all real attacks, how many the system found |
| F1-score | one value that combines precision and recall |
| ROC-AUC | how well the model separates the two classes at all thresholds |
| Macro F1 | the mean F1-score of the classes, each class with equal weight |
| Weighted F1 | the mean F1-score, each class weighted by its size |

Accuracy alone is not enough. Only 17% of the flows are attacks. A model that
calls everything benign gets 83% accuracy and finds no attack. Recall shows how
many attacks the system found.

## 11. Results

All results below come from the complete test set. No model saw this data
during training or during feature selection.

### Binary Intrusion Detection: benign or attack (499,596 test rows)

| Algorithm | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.9976 | 0.9868 | 0.9994 | 0.9930 | 0.99996 |
| Decision Tree | 0.9976 | 0.9871 | 0.9992 | 0.9931 | 0.9995 |
| K-Nearest Neighbors | 0.9903 | 0.9689 | 0.9746 | 0.9717 | 0.9954 |

Random Forest and Decision Tree give almost the same result. Random Forest
makes 1,194 errors and Decision Tree makes 1,175 errors. The difference is 19
flows of 499,596, so this project does not claim that one of the two is more
accurate.

Random Forest stays the main model for two reasons. It has the higher recall,
so it misses fewer attacks. It also has the higher ROC-AUC (0.99996 against
0.9995), which shows a better separation of the two classes at every threshold.

K-Nearest Neighbors is clearly weaker. It also trains on 80 times fewer rows,
so this comparison is not equal.

Source: `artifacts/models/model_results.csv`.

### Attack-Type Classification (85,145 attack rows)

| Metric | Value |
| --- | --- |
| Accuracy | 0.9994 |
| Macro precision | 0.9971 |
| Macro recall | 0.9852 |
| Macro F1 | 0.9907 |
| Weighted F1 | 0.9994 |

Per-class F1-score:

| Class | Test rows | F1-score |
| --- | --- | --- |
| FTP-Patator | 1,186 | 1.0000 |
| SSH-Patator | 644 | 1.0000 |
| Heartbleed | 2 | 1.0000 |
| DDoS | 25,603 | 0.9999 |
| DoS Hulk | 34,569 | 0.9997 |
| PortScan | 18,139 | 0.9996 |
| Bot | 390 | 0.9987 |
| DoS GoldenEye | 2,056 | 0.9966 |
| DoS Slowhttptest | 1,046 | 0.9933 |
| DoS slowloris | 1,075 | 0.9916 |
| Web Attack | 428 | 0.9860 |
| Infiltration | 7 | 0.9231 |

Source: `artifacts/attack_type_model/report.json`.

### The complete intrusion-detection system (499,596 test rows)

This is the result that a user of the system sees. Every test flow gets one
final label: `BENIGN` or one attack class.

| Metric | Value |
| --- | --- |
| Accuracy | 0.9975 |
| Macro F1 | 0.9436 |
| Weighted F1 | 0.9979 |
| Wrong final labels | 1,233 of 499,596 |
| Flows identified as attacks by binary intrusion detection | 86,229 |

The complete system accuracy is a little lower than the binary-detection
accuracy. Two errors are possible: binary intrusion detection can give the
wrong answer, and attack-type classification can give the wrong attack name.
For flows correctly detected as attacks, the attack-type classifier gives the
correct name in 99.95% of the cases.

The macro F1 of the system (0.9436) is lower than the macro F1 of attack-type
classification alone (0.9907). The reason is the `BENIGN` class, which now
takes part in the mean, together with the small classes.

Source: `artifacts/evaluation/complete_intrusion_detection_report.json`.

### Note about the target of 99.61%

The task named an expected accuracy of about 99.61%. This project measures
**99.76%** for binary intrusion detection with Random Forest. The two numbers are close, but they
do not come from the same experiment, so they are not directly comparable. The
result here depends on these choices:

- Duplicate flows are removed before the split. This lowers the row count from
  2,830,743 to 2,497,980 and removes an easy source of a high score.
- The split is stratified by attack label with the seed 42.
- The models use the 20 features of this project, after the correlation step.

A different preprocessing choice gives a different number. This project does
not tune any setting to reach a target value.

Table 4 of the CIC-IDS2017 paper reports the weighted average of seven
algorithms. For Random Forest it gives 0.98 precision, 0.97 recall, and 0.97
F1. For K-Nearest Neighbors it gives 0.96 / 0.96 / 0.96. The paper does not use
the sequential binary-detection and attack-type-classification design of this project, so these values are a general comparison
only. The results of this project are in the same range.

## 12. Model explainability

Notebook `06_model_explainability.ipynb` explains both Random Forest models
with three methods:

1. **Built-in importance.** This shows the features that the trees use most.
   The method is fast, but it can prefer a feature with many different values.
2. **Permutation importance.** The notebook shuffles one feature and measures
   the F1-score that the model loses. A large loss means an important feature.
3. **SHAP.** This shows how each feature moves one single prediction up or
   down. The notebook makes a summary plot for all flows and a waterfall plot
   for one flow.

The notebook saves the plots and a summary table in `artifacts/explainability`.

The saved feature values are standardized. A value of 2.5 means "2.5 standard
deviations above the training mean". It does not mean 2.5 bytes. The notebook
uses the saved scaler to also show the original network values.

SHAP explains the behavior of the model. It does not prove that a feature is
the cause of an attack.

## 13. Repository structure

```text
CICIDS2017-ML-IDS/
├── notebooks/                        # The implementation. Run in this order.
│   ├── 00_explore_data.ipynb         # Look at the raw and processed data
│   ├── 01_load_and_clean_data.ipynb  # Cleaning, split, scaling, feature selection
│   ├── 02_feature_selection.ipynb    # Review the feature-selection results
│   ├── 03_train_models.ipynb         # Binary intrusion detection: RF, Decision Tree, KNN
│   ├── 04_attack_type_classification.ipynb  # Attack-type classification
│   ├── 05_model_explainability.ipynb        # Importance and SHAP
│   └── 06_verify_trained_models.ipynb       # Check the models, measure the system
├── dataset/
│   ├── CSVs/
│   │   ├── MachineLearningCSV/       # The eight input CSV files
│   │   └── GeneratedLabelledFlows/   # Flows with identifiers. Not used.
│   └── processed/                    # train.parquet, test.parquet, scaler, report
├── artifacts/
│   ├── feature_selection/            # Scores, selected features, plots
│   ├── models/                       # Binary-detection models, metrics, plots
│   ├── attack_type_model/            # Attack-type model, metrics, confusion matrix
│   ├── evaluation/                   # Complete intrusion-detection results
│   └── explainability/               # Importance plots and SHAP plots
├── pyproject.toml
├── REPORT_FA.md
└── README.md
```

The notebooks contain the complete implementation. Earlier duplicate Python
scripts and their obsolete outputs were removed to keep one source of truth.

## 14. Installation

You need Python 3.11 or a later version.

Install [uv](https://docs.astral.sh/uv/), then create the environment:

```powershell
uv sync --extra notebook --extra explainability
```

`--extra notebook` installs Jupyter. `--extra explainability` installs SHAP.
Both are necessary for the complete set of notebooks.

### Dependencies

| Package | Use |
| --- | --- |
| pandas | read and change the data |
| numpy | numeric calculation |
| scikit-learn | the machine-learning models and the metrics |
| pyarrow | read and write Parquet files |
| matplotlib | the plots |
| joblib | save and load the models |
| tqdm | the progress bars |
| jupyter | run the notebooks (extra `notebook`) |
| seaborn | additional plots (extra `notebook`) |
| shap | the SHAP explanations (extra `explainability`) |

## 15. Prepare the dataset

1. Open <https://www.unb.ca/cic/datasets/ids-2017.html> and download
   `MachineLearningCSV.zip`.
2. Unpack the archive.
3. Copy the eight CSV files into `dataset/CSVs/MachineLearningCSV/`.

The folder must contain these files:

```text
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Monday-WorkingHours.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
```

A file name that ends with `.pcap_ISCX.csv` is still a CSV file. It is not a
PCAP file.

## 16. Train the models

Start Jupyter:

```powershell
uv run --extra notebook jupyter lab
```

Then run the notebooks in this order. Use "Run All Cells" in each notebook.

| Order | Notebook | Result | Time |
| --- | --- | --- | --- |
| 1 | `00_complete_pipeline.ipynb` | `dataset/processed/` and `artifacts/feature_selection/` | 10 to 15 min |
| 2 | `03_train_models.ipynb` | `artifacts/models/` | 4 to 6 min |
| 3 | `04_attack_type_classification.ipynb` | `artifacts/attack_type_model/` | about 1 min |

The times are from a normal desktop computer with eight CPU threads. Your
computer can be faster or slower. Notebook 00 needs about 8 GB of free memory,
because it holds the complete dataset in memory.

Notebook 00 must run first. The other notebooks read its output files.

Notebooks `01_explore_data.ipynb` and `02_feature_selection.ipynb` only read
results. Run them at any time after notebook 00.

Notebook 00 has two switches at the top:

```python
RUN_PREPROCESSING = True
RUN_FEATURE_SELECTION = True
```

Set a switch to `False` to read the saved results instead of calculating them
again.

## 17. Run the evaluation

Notebook `05_verify_trained_models.ipynb` loads the saved models and measures
them again. It does not train anything. It shows:

- the comparison table of the three binary intrusion-detection models,
- the confusion matrix of each model,
- the attacks missed by binary intrusion detection, for each attack class,
- the attack-type classification results,
- the result of the complete intrusion-detection system,
- single predictions with the confidence of the model.

The notebook writes its results to `artifacts/evaluation/`.

Notebook `06_model_explainability.ipynb` shows why the models decide as they
do. Run it after notebook 05.

### Inference on one flow

The pipeline has no separate inference script. To classify new flows, do these
steps in a notebook:

1. Load `dataset/processed/standard_scaler.joblib` and use `transform` on the
   new records. The new records must have the same 77 columns in the same
   order.
2. Keep the 20 columns from `selected_features.json`.
3. Load `artifacts/models/random_forest.joblib` and call `predict`.
4. For each record with the answer 1, load
   `artifacts/attack_type_model/random_forest_attack_type.joblib` and call
   `predict`.

Section 4 of notebook 05 contains this code for the test data.

## 18. Known limitations

1. **Two classes are too small to measure.** The test set contains 2 Heartbleed
   records and 7 Infiltration records. Their scores change completely when one
   record changes. Do not use these scores as evidence.

2. **The results apply to one network.** The dataset comes from one testbed and
   one week. The models learn the behavior of that network. The scores on a
   different network can be much lower.

3. **`Destination Port` is used as a number.** The models can learn port
   ranges. A port number is a name, not a quantity, so this can be a problem on
   another network.

4. **One split, not cross-validation.** All results come from one split with
   the seed 42. The project does not report a confidence interval.

5. **The features are selected for binary intrusion detection.** Attack-type classification reuses the same 20
   features. A separate selection for each attack class, as Table 3 of the
   paper shows, could give a higher macro F1.

6. **The system detects known attack types only.** The models learn from
   labelled examples. A new attack that does not look like the training data
   can pass without an alert.

## 19. References

1. Sharafaldin, I., Lashkari, A. H., and Ghorbani, A. A. "Toward Generating a
   New Intrusion Detection Dataset and Intrusion Traffic Characterization."
   *Proceedings of the 4th International Conference on Information Systems
   Security and Privacy (ICISSP)*, 2018, pages 108-116.
   File: `New Intrusion Detection Dataset.pdf`.

2. Network Security course, final project description, 2026.
   File: `cns- FinalProject-S2026.pdf`.

3. CIC-IDS2017 dataset: <https://www.unb.ca/cic/datasets/ids-2017.html>

4. Pedregosa, F., et al. "Scikit-learn: Machine Learning in Python."
   *Journal of Machine Learning Research*, 12, 2011, pages 2825-2830.
