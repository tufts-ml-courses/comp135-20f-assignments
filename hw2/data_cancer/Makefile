download_data::
	mkdir -p raw_data && cd raw_data/ && wget https://www.mskcc.org/sites/default/files/node/4509/documents/decisioncurveanalysis.zip && unzip decisioncurveanalysis.zip

dataset:: *.csv

*.csv:: raw_data/dca.csv
	cd prep_code/ && python make_clean_dataset.py

