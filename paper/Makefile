.PHONY: apj

figures:
	rm -f *.eps
	rm -f trigger.pdf
	rm -f expert_form.pdf

	cd ../figures && make

	cp ../figures/roc.eps .
	cp ../figures/gallery.eps .
	cp ../figures/fp.eps .
	cp ../figures/bootstrap_neg.eps .
	cp ../figures/cluster.eps .
	cp ../figures/expert_*eps .
	cp ../figures/dist_*eps .
	cp ../figures/mwp.eps .
	cp ../figures/new_gallery.eps .
	cp ../figures/new_score.eps .
	cp ../figures/preprocess.eps .
	cp ../figures/l???.eps .
	cp ../figures/zones.eps .
	cp ../figures/expert_form.pdf .

	cp ../figures/mwp_survey_gallery.png .
	cp ../figures/uniform_survey_gallery.png .

	./topdf
	echo "WARNING: Convert trigger.pdf to EPS files! I will make a crappy version with pdf2eps. You should do it in Illustrator"
	cp ../figures/trigger.pdf .
	pdf2ps trigger.pdf
	ps2eps trigger.ps
	cp ../figures/schematic* .

	cd ../scripts && make
	cp ../scripts/prob_table.tex .
	cp ../scripts/prob_table_head.tex .
	con decision_schematic.pdf decision_schematic.eps

apj:
	rm -rf apj
	mkdir -p apj
	for x in `python figure_report.py`; do cp $$x apj; done
	cp *tex apj
	cd apj && tar -czf apj.tgz * && mv apj.tgz ../
