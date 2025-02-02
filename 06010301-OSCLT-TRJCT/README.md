Dear reader,

in order to obtain Figure 02 in the paper, just open "a02_PLOTTR_OSCLT.py" and run it.

The files that can be found in this folder are:

- a00_PMTERS_CONST.py: File that calculates some of the parameters of the paper, such as alpha, gamma, R, S, etc.
                       This file is used by the PRTCLE files.

- a02_PLOTTR_OSCLT.py: Script that should be run in order to obtain Figure 02 in our paper.
		       This plot defines some of the parameters of the particles as well as the numerical schemes, calculates the solutions and plots them.

- a03_FIELD0_00000.py: File with the abstract class of a generic flow field.

- a03_FIELD0_OSCLT.py: File with the flow field of the Oscillatory background from Prasath et al. and the parameters of our paper.

- a09_PRTCLE_IMEX4.py: Script that has the class of a particle that given any field can calculate its trajectory using the second order IMEX method from the paper. The update method in this class calculates the trajectory of the particle.

- a09_PRTCLE_OSCLT.py: Script that has the class of a particle that moves in the oscillatory field. The update method in this class calculates the trajectory of the particle.

- a09_PRTCLE_TRAPZ.py: Script that has the class of a particle that given any field can calculate its trajectory using the Trapezoidal Rule method from the paper (2nd order implicit method). The update method in this class calculates the trajectory of the particle.

The resulting plots are saved within the folder VISUAL_OUTPUT.
