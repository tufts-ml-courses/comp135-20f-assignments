 Dataset credit: A. Vickers, Memorial Sloan Kettering Cancer Center [[original link]](https://www.mskcc.org/sites/default/files/node/4509/documents/dca-tutorial-2015-2-26.pdf).

Each patient in our dataset has been biopsied (fyi: in this case a [biopsy](https://www.cancer.net/navigating-cancer-care/diagnosing-cancer/tests-and-procedures/biopsy) is a short surgical procedure that is painful but with virtually no lasting harmful effects) to obtain a direct "ground truth" label so we know each patient's actual cancer status (binary variable, 1 means "has cancer", 0 means does not, column name is `cancer` in the $y$ data files). We want to build classifiers to predict whether a patient likely has cancer from easier-to-get information, so we could avoid painful biopsies unless they are necessary. Of course, if we skip the biopsy, a patient with cancer would be left undiagnosed and therefore untreated. We're told by the doctors this outcome would be life-threatening.

Binary Outcomes are in `y_{split}.csv` files.
* 1 means cancer detected
* 0 means no cancer

Features are in `x_{split}.csv`
* `age` : patient age in years
* `famhistory` : binary indicator of family history of cancer
* `marker` : numerical score for  a new easy-to-measure biomarker

