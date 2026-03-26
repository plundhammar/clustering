List of files used to get the resutls shown in the ppt in the Luca previous work directory.

## reading_data ##

These code files are used to read data from ROOT files from allpix in two different cases. If the detector is not pixelated and is considered just a bulk of silicon then use the bulk_silicon one. wereas if it is pixelated and we simulate the full chain from the photons to the interactions and electron tracks we use the other one. These 2 are not the latest optimized versions but they still work.

## radii_distribution ##

This is used to get the radii distributions that will be later used in the grouping algorithms

## grouping_interactions ##

This is used to group the interactions, the other version with primary_radii is the same but we use the radii distribution characterized from only primary photon's interactions.

## analyze results ##

this should give the plots in the ppt from 03-31

## analyze synthesis ##

this should give the plots in the ppt from 09-29. the realizations variation gives instead the plots for the false positives.
