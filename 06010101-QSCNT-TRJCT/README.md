Dear reader,

in order to obtain Figure 03 in the paper, just open "a02_PLOTTR_QSCNT.py" and run it.

The files that can be found in this folder are:

- a00_MAT-F_VALUES.npy: Values of the Matrix M_F used in Prasath et al.'s method. Calculated automatically.

- a00_MAT-H_VALUES.npy: Values of the Matrix M_H used in Prasath et al.'s method. Calculated automatically.

- a00_MAT-I_VALUES.npy: Values of the Matrix M_I used in Prasath et al.'s method. Calculated automatically.

- a00_MAT-Y_VALUES.npy: Values of the Matrix M_Y used in Prasath et al.'s method. Calculated automatically.

- a00_PMTERS_CONST.py: File that calculates some of the parameters of the paper, such as alpha, gamma, R, S, etc.
                       This file is used by the PRTCLE files.

- a00_TSTEP_VALUE.npy: Values of the time vector used in Prasath et al's method. Calculated automatically.

- a02_PLOTTR_QSCNT.py: Script that should be run in order to obtain Figure 03 in our paper.
		       This plot defines some of the parameters of the particles as well as the numerical schemes, calculates the solutions and plots them.

- a03_FIELD0_00000.py: File with the abstract class of a generic flow field.

- a03_FIELD0_QSCNT.py: File with the flow field of the quiescent background from Prasath et al. and the parameters of our paper.

- a09_PRTCLE_DTCHE.py: Script that has the class of a particle that given any field can calculate its trajectory using any of Daitche's methods. The Euler, AdamBashf2, AdamBashf3 methods within this class calculate the trajectory of the particle with a first, second and third order method, respectively.

- a09_PRTCLE_PRSTH.py: Script that has the class of a particle that given any field can calculate its trajectory using Prasath et al.'s method. The update method in this class calculates the trajectory of the particle.

- a09_PRTCLE_QSCNT.py: Script that has the class of a particle that moves in the quiescent field. The update method in this class calculates the trajectory of the particle.

The resulting plots are saved within the folder VISUAL_OUTPUT.
