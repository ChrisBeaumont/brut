figures:
	rm -f *.eps
	rm -f trigger.pdf

	cd ../figures && make

	cp ../figures/roc.eps .
	cp ../figures/gallery.eps .
	cp ../figures/fp.eps .
	cp ../figures/bootstrap_neg.eps .
	cp ../figures/cluster.eps .
	cp ../figures/cluster_confusion.eps .
	cp ../figures/expert_*eps .
	cp ../figures/dist_*eps .
	cp ../figures/mwp.eps .
	cp ../figures/new_gallery.eps .
	cp ../figures/new_score.eps .
	cp ../figures/l???.eps .

	./topdf
	cp ../figures/trigger.pdf .
	pdf2ps trigger.pdf
	ps2eps trigger.ps
	cp ../figures/schematic* .
