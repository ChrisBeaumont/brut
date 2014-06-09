"""
Regrid data to common grid
"""
import os
from glob import glob
import logging
from subprocess import check_call

from bubbly.util import lon_offset, up_to_date

def register(files, lcen, outfile, clobber=False):
    """Regrid a list of files to a 2x2deg tile, with 2" pixels

    Parameters
    ----------
    files: List of files to combine
    lcen: Desired longitude of map center
    outfile: Path to output file
    clobber: If True, overwrite existing files

    Regardless of clobber, If the output file exists and is more up to
    date than the input files, nothing happens
    """
    if os.path.exists(outfile) and not clobber:
        logging.info("File exists and clobber=False. Skipping. %s" % outfile)
        return

    if up_to_date(files, outfile):
        logging.info("%s is up to date" % outfile)
        return

    if len(files) == 0:
        logging.info("Input file list empty. Skipping longitude %i" % lcen)
        return

    params = dict(CELESTIAL_TYPE='Galactic',
                  PROJECTION_TYPE='CAR',
                  CENTER_TYPE='MANUAL',
                  CENTER = '%f,0.0' % lcen,
                  PIXEL_SCALE = '2.0',
                  PIXELSCALE_TYPE = 'MANUAL',
                  IMAGE_SIZE = "7200,3600",
                  SUBTRACT_BACK='N',
                  IMAGEOUT_NAME=outfile,
                  WRITE_FILEINFO='Y',
                  WRITE_XML='N',
                  )
    params = ' '.join('-%s %s' % (k, v) for k, v in params.items())
    cmd = 'swarp %s %s' % (' '.join(files), params)
    print cmd
    check_call(cmd.split())


def match_and_register(f_lon_map, lon, lon_thresh, out_template, clobber):
    files = [k for k, v in f_lon_map.items() if
             lon_offset(v, lon) < lon_thresh]
    register(files, lon, out_template % lon, clobber)


def register_irac_3(lon,
                    data_dir='galaxy/raw/',
                    out_dir='galaxy/registered/',
                    clobber=False
                    ):
    """
    Create IRAC tiles at a given longitude

    Parameters
    ----------
    lon: The longitude tiles to make
    data_dir: Directory of the input data
    out_dir: Directory to save registered data
    clobber: True to overwrite existing out of data data. False to skip
    """
    offset = len(data_dir)
    i3 = glob(data_dir + 'GLM*_I3.fits')
    f3_lon_map = { f: int(f[offset + 4 : offset + 7]) for f in i3}
    match_and_register(f3_lon_map, lon, 4.5,
                       out_dir + '%3.3i_i3.fits', clobber)

def register_irac_4(lon,
                    data_dir='galaxy/raw/',
                    out_dir='galaxy/registered/',
                    clobber=False
                    ):
    """ Equivalent of register_irac_3 for irac_4 data """
    offset = len(data_dir)
    i4 = glob(data_dir + 'GLM_*I4.fits')
    f4_lon_map = { f: int(f[offset + 4 : offset + 7]) for f in i4}
    match_and_register(f4_lon_map, lon, 4.5,
                       out_dir + '%3.3i_i4.fits', clobber)

def register_mips(lon,
                  data_dir='galaxy/raw/',
                  out_dir='galaxy/registered/',
                  clobber=False
                  ):
    """The equivalent of register_irac for mips data"""


    mips = glob(data_dir + 'MG*fits')
    offset = len(data_dir)
    f_lon_map = {f: int(f[offset + 2:offset + 5]) for f in mips}
    match_and_register(f_lon_map, lon, 4.5,
                       out_dir + '%3.3i_mips.fits', clobber)

def register_heatmaps(lon,
                      data_dir='galaxy/mwp_heatmaps/',
                      out_dir='galaxy/registered/',
                      clobber=False):
    """ The equivalent of register_irac for MWP heatmaps """
    heat = glob(data_dir+'*_float.fits')
    f_lon_map = {h : int(h.split('/')[-1].split('_')[0]) for h in heat}

    match_and_register(f_lon_map, lon, 4.5,
                       out_dir + '%3.3i_heatmap.fits', clobber)

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Register raw data to tiles')
    parser.add_argument('lon',
                        type=int)
    parser.add_argument('data',
                        choices=['i3', 'i4', 'mips', 'heat'],
                        help='Which data to register')
    parser.add_argument('--clobber', action='store_true',
                        help='Overwrite out-of-date files')

    args = parser.parse_args()

    func = {'i3' : register_irac_3,
            'i4' : register_irac_4,
            'mips' : register_mips,
            'heat' : register_heatmaps}
    func[args.data](args.lon, clobber=args.clobber)
