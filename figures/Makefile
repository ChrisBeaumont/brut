all : roc.eps gallery.eps fp.eps bootstrap_neg.eps

roc.eps : roc_plot.py
	python roc_plot.py

%.eps : %.py
	python $*.py
