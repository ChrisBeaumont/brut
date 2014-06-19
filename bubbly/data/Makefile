all: scrape register

# note: these require local data to make, so are included in repo
pickle: catalog.pkl hiq.pkl


%.pkl: make_%.py
	python make_$*.py

scrape: scrape.py
	python scrape.py

# These lines always issue python commands to build tiles.
# Python does checking whether files are up to date or not
# make register will run everything
LONS := $(shell seq 0 360)

tile_i3 :=$(addsuffix _r_i3,${LONS})
tile_i4 :=$(addsuffix _r_i4,${LONS})
tile_mips :=$(addsuffix _r_mips,${LONS})

register: ${tile_i3} ${tile_i4} ${tile_mips}

#by 'filter'
register_i3: $(tile_i3)
register_i4: $(tile_i4)
register_mips: $(tile_mips)

%_r_i3:
	python register.py $* i3 --clobber

%_r_i4:
	python register.py $* i4 --clobber

%_r_mips:
	python register.py $* mips --clobber
