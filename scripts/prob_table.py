from astropy.io import ascii
from bubbly.dr1 import get_catalog
from astropy.table import Column

hr = get_catalog()[:, -1]

data = ascii.read('../data/pdr1.csv')
caption=r"""Bubble probabilities for objects in the Milky Way Project catalog. Non-finite probabilities correspond to objects near the edge of the data, which Brut cannot analyze.\label{table:prob_table}"""

data['a'] *= 60
data['b'] *= 60
data['t'] *= 60
hr = Column(name='Hit Rate', data=hr)

data.add_column(hr, index=6)

data.rename_column('lon', '$\ell$')
data.rename_column('lat', '$b$')
data.rename_column('pa', 'PA')
data.rename_column('a', 'Semimajor axis')
data.rename_column('b', 'Semiminor axis')
data.remove_columns(['t'])
data.rename_column('prob', 'P(Bubble)')

formats = {'P(Bubble)': "%0.3f",
           'PA': '%i',
           'Semimajor axis': '%0.1f',
           'Semiminor axis': '%0.1f',
           'Thickness': '%0.1f'}

units = {'$\ell$': 'deg',
         '$b$':'deg',
         'PA':'deg',
         'Semimajor axis': 'arcmin',
         'Semiminor axis': 'arcmin',
         'Thickness' : 'arcmin',
         }

ascii.write(data, 'prob_table.tex', Writer=ascii.latex.AASTex,
            col_align='rrrrrrr', caption=caption,
            latexdict={'units':units},
            formats=formats
            )

data = data[0:10]
ascii.write(data, 'prob_table_head.tex', Writer=ascii.latex.AASTex,
            col_align='rrrrrrr', caption=caption,
            latexdict={'units':units},
            formats=formats
            )
