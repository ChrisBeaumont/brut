all : roc.eps gallery.eps fp.eps bootstrap_neg.eps roc_all.eps cluster.eps cluster_confusion.eps expert_mwp_score.eps expert_mwp_votes.eps expert_roc.eps expert_uniform_score.eps expert_uniform_votes.eps mwp.eps dist_lon.eps dist_lat.eps dist_hii_score.eps l317.eps l305.eps l299.eps new_gallery.eps new_score.eps trigger.eps

dist_%.eps: distributions.py ../notebooks/distributions.ipynb
	python distributions.py

expert_%.eps : expert_votes.py ../notebooks/Expert_Votes.ipynb
	python expert_votes.py

l317.eps l305.eps l299.eps : wide_area_plots.py ../notebooks/wide_area_plots.ipynb
	python wide_area_plots.py

roc.eps : roc_plot.py
	python roc_plot.py

%.eps : %.py
	python $*.py

new_%.eps: new_gallery.py ../notebooks/full_search_distributions.ipynb
	python new_gallery.py

trigger.eps: trigger.py ../notebooks/trigger.ipynb
	python trigger.py
