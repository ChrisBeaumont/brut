all: pdr1.csv known_catalog_hrds_v1.0.txt hrds_table1.txt hrds_distances.txt

pdr1.csv: probabilistic_dr1.py
	python probabilistic_dr1.py

known_catalog_hrds_v1.0.txt:
        wget http://www.cv.nrao.edu/hrds/$@

hrds_table1.txt:
        wget http://www.cv.nrao.edu/hrds/$@

hrds_distances.txt:
        wget http://www.cv.nrao.edu/hrds/$@
