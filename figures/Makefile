all : roc.eps gallery.eps fp.eps bootstrap_neg.eps roc_all.eps cluster.eps cluster_confusion.eps expert_mwp_score.eps expert_mwp_votes.eps expert_roc.eps expert_uniform_score.eps expert_uniform_votes.eps

expert_%.eps : expert_votes.py ../notebooks/Expert_Votes.ipynb
	python expert_votes.py

roc.eps : roc_plot.py
	python roc_plot.py

%.eps : %.py
	python $*.py
