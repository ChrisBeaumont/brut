"""
This script runs the cluster/yso correlation described in
Kendrew et al 2012. It looks for overdensities of YSO's along
bubble rims
"""
from astropy.io import ascii
from astropy.table import Table
import numpy as np

from calc_corr import calc_corr


def get_bubbles():
    """ Read the bubble catalog"""
    bubble_file = '../data/pdr1.csv'
    bubbles = ascii.read(bubble_file, delimiter=',', data_start=1)
    reff = np.sqrt(bubbles['a'] * bubbles['b']) * 60
    reff.name = 'reff'
    bubbles.add_column(reff)

    bubbles['lon'][bubbles['lon'] > 180] -= 360
    #remove bubbles with |l| < 10, which YSO cat doesn't cover
    bubbles = bubbles[np.abs(bubbles['lon']) >= 10]

    return bubbles


def get_ysos():
    """Read the YSO catalog"""
    yso_file = '../data/rms_allyoung_full_nokda.csv'
    columns = ['id', 'rmsid', 'name', 'type', 'rahex',
               'dechex', 'flux8', 'flux12', 'flux14',
               'flux21', 'jmag', 'hmag', 'kmag', 'vlsr',
               'rgc', 'kds', 'd', 'blank', 'firlum', 'firflux',
               'blank2', 'lon', 'lat']
    skip = ['id', 'rahex', 'dechex', 'blank', 'firflux', 'blank2']
    ysos = ascii.read(yso_file, delimiter=',', names=columns,
                      exclude_names=skip)

    #filter YSOs outside the survey area of Bubble Catalog
    ysos['lon'][ysos['lon'] > 180] -= 360
    ysos['type'] = np.char.rstrip(ysos['type'], '?')
    keep = (np.abs(ysos['lat']) <= 1) & (np.abs(ysos['lon']) <= 65)
    ysos = ysos[keep]

    return ysos


def run_correlation(bubble, yso, outfile, binStep=0.2):
    """Cross correlate and write to file

    Parameters
    ----------
    bubble : astropy Table
    yso : astropy Table
    outfile : str
    """
    theta, corr, err = calc_corr(bubble, yso, corrType='x',
                                 rSize=50, nbStrap=100, binStep=binStep)
    t = Table([theta, corr, err], names=['theta', 'w', 'dw'])
    t.write(outfile, format='ascii', delimiter=',')


def shuffle(table):
    result = Table(table)
    result['lon'] = np.random.permutation(table['lon'])
    result['lat'] = np.random.permutation(table['lat'])
    result['reff'] = np.random.permutation(table['reff'])
    return result


def main():
    from os import path
    np.random.seed(42)

    out_dir = path.abspath(path.join(path.dirname(__file__), '..', 'data'))
    bubbles = get_bubbles()
    ysos = get_ysos()
    ysos_sub = ysos[np.in1d(ysos['type'], ['HII/YSO', 'YSO'])]

    b1 = bubbles[bubbles['prob'] < .5]
    b2 = bubbles[(bubbles['prob'] >= .5) & (bubbles['prob'] < .9)]
    b3 = bubbles[bubbles['prob'] >= .9]

    s, m, l = [bubbles[bubbles['reff'] > thresh]
               for thresh in np.percentile(bubbles['reff'], [50, 75, 90])]

    #randomly reposition bubbles to overlap stars
    ind = np.random.randint(0, len(ysos) - 1, len(bubbles))
    sb = Table(bubbles)
    sb['lon'] = ysos['lon'][ind]
    sb['lat'] = ysos['lat'][ind]
    run_correlation(sb, ysos, path.join(out_dir, 'cluster_ysopos.csv'))

    # baseline
    run_correlation(bubbles, ysos, path.join(out_dir, 'cluster_all.csv'))
    run_correlation(b1, ysos, path.join(out_dir, 'cluster_plow.csv'))
    run_correlation(b2, ysos, path.join(out_dir, 'cluster_pmid.csv'))
    run_correlation(b3, ysos, path.join(out_dir, 'cluster_phi.csv'))

    # shuffled
    run_correlation(shuffle(bubbles), ysos,
                    path.join(out_dir, 'cluster_all_s.csv'))
    run_correlation(shuffle(b1), ysos,
                    path.join(out_dir, 'cluster_plow_s.csv'))
    run_correlation(shuffle(b2), ysos,
                    path.join(out_dir, 'cluster_pmid_s.csv'))
    run_correlation(shuffle(b3), ysos,
                    path.join(out_dir, 'cluster_phi_s.csv'))

    # no HII regions in yso catalog
    run_correlation(bubbles, ysos_sub,
                    path.join(out_dir, 'cluster_all_nohii.csv'))
    run_correlation(b1, ysos_sub,
                    path.join(out_dir, 'cluster_plow_nohii.csv'))
    run_correlation(b2, ysos_sub,
                    path.join(out_dir, 'cluster_pmid_nohii.csv'))
    run_correlation(b3, ysos_sub,
                    path.join(out_dir, 'cluster_phi_nohii.csv'))

    # binned by size
    run_correlation(s, ysos, path.join(out_dir, 'cluster_small.csv'))
    run_correlation(m, ysos, path.join(out_dir, 'cluster_medium.csv'))
    run_correlation(l, ysos, path.join(out_dir, 'cluster_large.csv'))

    #different bins
    run_correlation(bubbles, ysos, path.join(out_dir, 'cluster_all_bin.csv'),
                    binStep=0.1)
    run_correlation(b1, ysos, path.join(out_dir, 'cluster_plow_bin.csv'),
                    binStep=0.1)
    run_correlation(b2, ysos, path.join(out_dir, 'cluster_pmid_bin.csv'),
                    binStep=0.1)
    run_correlation(b3, ysos, path.join(out_dir, 'cluster_phi_bin.csv'),
                    binStep=0.1)




if __name__ == '__main__':
    main()
