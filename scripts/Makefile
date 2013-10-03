
TARGETS := ../models/full_classifier.dat ../models/benchmark_scores.json ../models/l035_scores.json ../models/bubble_scores.json ../models/l035_small_scores.json ../models/random_scores.json ../models/l305_scores.json ../data/cluster_plow.csv ../data/cluster_pmid.csv ../data/cluster_phi.csv ../data/cluster_ysopos.csv prob_table.tex

MODEL := ../models/full_classifier.dat

LONS := $(shell echo `seq -f%03g 0 68` `seq -f%03g 292 360`)


all : $(TARGETS)

search: search_submit search_fetch

search_fetch : $(addprefix ../data/full_search/, $(addsuffix .h5, ${LONS}))
search_submit: $(addprefix submit_, ${LONS})
search_merge: ../data/full_search.csv

../models/full_classifier.dat : build_full_classifier.py
	python build_full_classifier.py

../models/benchmark_scores.json: benchmark_classifications.py $(MODEL)
	python benchmark_classifications.py

../models/l035_scores.json: l035_classify.py $(MODEL)
	python l035_classify.py

../models/l035_small_scores.json: l035_small_classify.py $(MODEL)
	python l035_small_classify.py

../models/l305_scores.json: l305_classify.py $(MODEL)
	python l305_classify.py

../models/bubble_scores.json: bubble_classify.py $(MODEL)
	python bubble_classify.py

../models/random_scores.json: random_classifications.py $(MODEL)
	python random_classifications.py

../data/cluster%.csv: trigger.py
	pythonw trigger.py

../data/full_search/%.h5: classify_all.py $(MODEL)
	python classify_all.py fetch $*

submit_%:
	python classify_all.py submit $*

../data/full_search.csv: merge_full_search.py
	python merge_full_search.py

prob_table.tex: prob_table.py
	python prob_table.py
