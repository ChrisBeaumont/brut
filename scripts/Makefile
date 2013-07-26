
TARGETS := ../models/full_classifier.dat ../models/benchmark_scores.json ../models/l035_scores.json ../models/bubble_scores.json ../models/l035_small_scores.json

MODEL := ../models/full_classifier.dat

all : $(TARGETS)

../models/full_classifier.dat : build_full_classifier.py
	python build_full_classifier.py

../models/benchmark_scores.json: benchmark_classifications.py $(MODEL)
	python benchmark_classifications.py

../models/l035_scores.json: l035_classify.py $(MODEL)
	python l035_classify.py

../models/l035_small_scores.json: l035_small_classify.py $(MODEL)
	python l035_small_classify.py

../models/bubble_scores.json: bubble_classify.py $(MODEL)
	python bubble_classify.py
