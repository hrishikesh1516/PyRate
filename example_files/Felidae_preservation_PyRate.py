#!/usr/bin/env python
# PyRate input file for the Felidae preservation example
# Generated from Felidae_preservation_example.txt (1 replicate, ages = midpoint of occurrence range)
# Taxa: 12 Felidae lineages (9 extinct, 3 extant) spanning ~25–0 Ma
# Use this file to explore preservation (sampling) models in PyRate.
# See tutorials/pyrate_tutorial_preservation.md for step-by-step instructions.
from numpy import * 

# Each array contains the sampled fossil ages (Ma) for one lineage.
# Ages are drawn from the midpoint of the min_age/max_age range in the source .txt file.

data_1=[
array([21.73, 21.02, 21.73]),                                     # Proailurus_lemanensis
array([14.32, 15.0, 13.80, 15.41, 12.63]),                        # Pseudaelurus_quadridentatus
array([9.82, 10.32, 9.25, 10.57]),                                 # Metailurus_major
array([8.48, 9.32, 7.17, 7.5, 6.42, 8.0]),                        # Machairodus_aphanistus
array([7.52, 8.35, 6.67, 7.75]),                                   # Paramachairodus_orientalis
array([2.9, 3.97, 2.65, 3.1]),                                     # Dinofelis_barlowi
array([1.35, 1.15, 0.76, 0.96, 1.44]),                            # Homotherium_serum
array([0.91, 1.44, 0.36, 0.56, 0.21, 1.05]),                      # Smilodon_fatalis
array([1.25, 1.1, 0.51, 0.75]),                                    # Panthera_gombaszoegensis
array([0.41, 0.16, 0.5, 0.07, 0.31]),                             # Panthera_leo
array([0.9, 0.35, 0.65, 0.20]),                                    # Panthera_tigris
array([1.64, 0.36, 0.81, 1.44]),                                   # Acinonyx_jubatus
]

d = [data_1]
names = ['Felidae_preservation_1']
def get_data(i): return d[i]
def get_out_name(i): return names[i]

taxa_names = [
    'Proailurus_lemanensis',
    'Pseudaelurus_quadridentatus',
    'Metailurus_major',
    'Machairodus_aphanistus',
    'Paramachairodus_orientalis',
    'Dinofelis_barlowi',
    'Homotherium_serum',
    'Smilodon_fatalis',
    'Panthera_gombaszoegensis',
    'Panthera_leo',
    'Panthera_tigris',
    'Acinonyx_jubatus',
]
def get_taxa_names(): return taxa_names
