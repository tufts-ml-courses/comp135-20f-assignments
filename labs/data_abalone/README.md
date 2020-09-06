Your ecologist colleagues from Australia have given you a dataset of physiological measurements related to abalone, an abundant shellfish [[Wikipedia article on abalone](https://simple.wikipedia.org/wiki/Abalone)].

Ecologists are interested in monitoring abalone population health by tracking various measurements (length, weight) of these creatures, as well as their age. While the physical measurements are somewhat easy to obtain in the field, directly measuring age is a boring and time-consuming task (cut open the shell, stain it, count the number of rings on the sheel visible through a microscope). The age is known to be equal to 1.5 plus the number of rings.

You have been asked to build an *ring count* predictor for abalone, which is naturally a **regression** problem. You'll have the following input measurements for each abalone:


| column name      | type    | unit | description |
| ---------------- | ------- | ---- | ----------- |
| is_male          | binary  |      | 1 = 'male', 0 = 'female'
| length_mm        | numeric | mm   | longest shell measurement
| diam_mm          | numeric | mm   | diameter of shell, perpendicular to length
| height_mm        | numeric | mm   | height of shell (with meat inside)
| whole_weight_g   | numeric | gram | entire creature weight (shell + guts + meat)
| shucked_weight_g | numeric | gram | weight of the meat
| viscera_weight_g | numeric | gram | weight of the guts (after bleeding)
| shell_weight_g   | numeric | gram | weight of shell alone (after drying)


If you like, you can browse the web to see [visually what meat, guts, and shells look like](
https://www.thespruceeats.com/how-to-clean-abalone-2216416).

In this folder, we have provided a predefined train/validation/test split of this dataset, stored on-disk in comma-separated-value (CSV) files:

* Training set: x_train.csv, y_train.csv,
* Validation set: x_valid.csv, y_valid.csv
* Test set: x_test.csv, and y_test.csv.


# References

Source for this dataset:
<https://archive.ics.uci.edu/ml/datasets/abalone>