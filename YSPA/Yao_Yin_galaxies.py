import numpy as np
import matplotlib.pyplot as plt


galaxy_data = np.genfromtxt('galaxies.dat',skip_header=1)
ra, dec, ve = galaxy_data.T

ra_ge_180 = ra>=180
dec_south = dec <0
