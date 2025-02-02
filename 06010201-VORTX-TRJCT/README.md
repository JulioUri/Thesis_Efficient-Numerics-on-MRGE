Dear reader,

in order to obtain Figure 01 in the paper, just open "a02_PLOTTR_ANALY.py" and run it.

The files that can be found in this folder are:

- a00_PMTERS_CONST.py: File that calculates some of the parameters of the paper, such as alpha, gamma, R, S, etc.
                       This file is used by the PRTCLE files.

- a02_PLOTTR_ANALY.py: Script that should be run in order to obtain Figure 01 in our paper.
		       This plot defines some of the parameters of the particles as well as the numerical schemes, calculates the solutions and plots them.

- a03_FIELD0_00000.py: File with the abstract class of a generic flow field.

- a03_FIELD0_ANALY.py: File with the flow field of the Vortex from Candelier et al. The name of the file "ANALY" refers to the fact that there is an analytic solution, which was the only one we knew at the beginning.

- a09_PRTCLE_ANALY.py: Script that has the class of a particle that moves in the vortex field. The update method in this class calculates the trajectory of the particle.

- a09_PRTCLE_DIRK4.py: Script that has the class of a particle that given any field can calculate its trajectory using the DIRK method from the paper. The update method in this class calculates the trajectory of the particle.

- a09_PRTCLE_IMEX4.py: Script that has the class of a particle that given any field can calculate its trajectory using the fourth order IMEX method from the paper. The update method in this class calculates the trajectory of the particle.

The resulting plots are saved within the folder VISUAL_OUTPUT.
