# Code repository for the Ph.D. Thesis of Julio Urizarna-Carasa

This repository contains the Python scripts implemented for the development of the Ph.D. thesis of Julio Urizarna-Carasa _An efficient numerical implementation for the Maxey-Riley-Gatignol Equation_. This code can reproduce all figures shown in the thesis's paper. You can freely use this code in accordance with its [license](https://github.com/JulioUri/Thesis_Efficient-Numerics-on-MRGE/blob/main/LICENSE). However, if you use this code or parts of it for a publication, please make sure to cite both this repository and the paper.

For questions, contact [M. Sc. Julio Urizarna-Carasa](https://www.linkedin.com/in/julio-urizarna/) or [Prof. Dr. Daniel RUPRECHT](https://www.mat.tuhh.de/home/druprecht/?homepage_id=druprecht).


## How to run the code

> :mega: Note that I do not hold ownership on the file `00_2mm_Faraday_50Hz_40ms_1_6g.mat` that contains the data with the experimental field. Please contact [Prof. Dr. Alexandra VON KAMEKE](https://www.haw-hamburg.de/hochschule/beschaeftigte/detail/person/person/show/alexandra-von-kameke/) if you are interested in using it.

In order to reproduce the figures/tables/data from the thesis yourself, you will need to find the appropriate script to run. Below you will find a fast guide of how to otain the data without having to dig deep into the software.

In this repository you will find many folders that contain the necessary code to reproduce the figures or tables from the thesis paper. The name of the folder follows the following naming convenction:

- First, a code of eight numbers refering to the chapter, section, subsection and figure/table ordering, i.e. the code 06020107 implies that the figure or table is presented in Chapter 6, section 2, subsection 1 and it is the figure/table number 7 within that subsection. The last two folders, 07020000 and 07030000 do not exactly follow that naming convenction, since the data generated with those scripts covers more than one subsection and many figures and tables.
- Second, there is a code of one or words with a description of what the information it can be obtained with the code inside. Below you can find a dictionary with the meaning of the words used:
  * SPARS-MATRX: Code that obtains the image of the Sparse matrix.
  * QSCNT: Refers to code that obtains information for the quiescent flow field.
  * VORTX: Refers to code that obtains information for the vortex flow field.
  * OSCLT: Refers to code that obtains information for the oscillatory flow field.
  * DGYRE: Refers to code that obtains information for the double gyre flow field.
  * BICKL: Refers to code that obtains information for the Bickley jet flow field.
  * DATA1: Refers to code that obtains information for the Faraday flow, also referred as the experimental flow field.
  * TRJCT: Code that obtained trajectory data of a particle in a flow field.
  * CNVRG-R: Code that gets information (figures, tables and raw data) about the convergence of the methods as the parameter R varies.
  * CNVRG-S: Code that gets information (figures, tables and raw data) about the convergence of the methods as the parameter S varies.
  * PRCSN: Code that plots the work-precision plot of the methods in a specific flow field
  * C: Code that plots the convergence with respect to the parameter "c" of the FD methods in a specific flow field.
  * DIFFR: Scripts that obtain the difference between solutions of particle trajectories calculated with and without Basset History Term.
  * RDIST: Scripts that obtain the radial distance of particle trajectories calculated with and without Basset History Term.
  * CLUST: Scripts that generate trajectory data for clusters of particles. This is done using parallelisation.
  * FTLE: Scripts that generate the Finite Time Lyapunov Exponents (FTLE), given the particle trajectories from the CLUST folder.

Within each folder, there is a file with either the prefixes "a02" (a02_PLOTTR_) or "a06" (either a06_CONVRG_ or a06_PRECSN_ or a06_ERRORS_). These are the scripts that must be run in order to obtain the figures/tables/data:

- a02_PLOTTR_ files produce trajectory data,
- a06_CONVRG_ files produce Convergence plots or tables,
- a06_PRECSN_ files generate Work-Precision plots and
- a06_ERRORS_ file is only used once to obtain the error for different values of c for a particle in the Vortex. 

Within the files above, some parameters can be modified, such as time steps, initial and final times, particle and fluid densities, particle radius, etc.

All other files are supporting material used to run the scripts:

- a00_PMTERS_CONST.py defines the constants of the MRE (R, S, alpha, etc.),
- a00_MAT and a00_TSTEP define the matrices and time grid associated to Prasath et al.'s solver (enable a speed-up of the calculations),
- a03_FIELD0_ files define the velocity fields,
- a09_PRTCLE_ files hold the numerical or analytical solver (in case an analytical solution is available).

Within the OUTPUT folders, one can find the data associated to each Plot and Table as well as the Plot and Table themselves. Generating a single Figure or Table takes between 1 to 5 hours. The reader may therefore prefer to copy the data from the .txt file and generate the figures from it.

The toolbox currently depends on classical Python packages such as [Numpy](https://numpy.org/), [SciPy](https://scipy.org/) and [Matplotlib](https://matplotlib.org/).

Additional libraries like [progressbar2](https://pypi.org/project/progressbar2/) may also require installation.

The Anaconda environment used during the developing process can be found in the file *base_environment.yml*. 

### Script naming convention

Each .py file in the folders is there for a reason. If you wish to change the code yourself, you will need a bit of knowledge about each of the files there. Each file is made up of a code with the following structure *z99_AAAAAA_BBBBB.py*. Prefix *z99* is linked to the code *AAAAAA*, but enables an appropiate sorting within the folder different from Alphabetical sorting. Code *AAAAAA* summarizes what the code does:

 - either defines parameters (PMTERS),
 - plots trajectories (PLOTTR),
 - creates Convergence or Work-precision plots (CONVRG, ERRORS or PRECSN),
 - defines the velocity fields (FIELD0) or
 - the particle classes associated to each solver (PRTCLE)

 The second part of the code, i.e. *BBBBB*, provides more specific information about the file, such as the type of solver or type of velocity field:

 - CONST stands for the fact that the file defines Constant parameters,
 - ANALY stands for the Analytical solution of the Vortex (at the time of developing this code we were only aware of one analytical solution),
 - OSCLT stands for oscillatory background, either its analytical solution (PRTCLE_OSCLT) or the field (FIELD0_OSCLT),
 - QSCNT stands for quiescent flow, either its analytical solution (PRTCLE_QSCNT) or the zero field (FIELD0_QSCNT),
 - VORTX stands for vortex described by Candelier et al. (2004), either its analytical solution (PRTCLE_VORTX) or the field (FIELD0_VORTX),
 - BICKL stands for Bickley Jet,
 - DATA1 stands for the Faraday flow, obtained from experimental Data,
 - 00000 stands for the abstract particle class,
 - TRAPZ holds the Trapezoidal Rule solver,
 - DTCHE holds Daitche's 1st, 2nd, 3rd order schemes,
 - PRSTH holds Prasath et al.'s solver,
 - DIRK4 holds the solver with the 4th order DIRK method,
 - IMEX4 holds the solver with either the 1st, 2nd, 3rd and 4th order IMEX.

## Aditional considerations

It is important to note that the version of the MRGE used here corresponds to its nondimensional form and therefore **all velocities, distances and times taken as an input must be NONDIMENSIONAL**.

## Acknowledgements

This repository is a result of the work carried out by 
[ M. Sc. Julio URIZARNA-CARASA](https://www.mat.tuhh.de/home/jurizarna_en) and [Prof. Dr. Daniel RUPRECHT](https://www.mat.tuhh.de/home/druprecht/?homepage_id=druprecht), from Hamburg University of Technology.

<p align="center">
  <img src="./Logos/tuhh-logo.png" height="55"/> &nbsp;&nbsp;&nbsp;&nbsp;
</p>

We are grateful to [Prof. Dr. Alexandra VON KAMEKE](https://www.haw-hamburg.de/hochschule/beschaeftigte/detail/person/person/show/alexandra-von-kameke/) for providing us with the data from the experimental measurements of the Faraday flow and to [Prof. Dr. Kathrin PADBERG-GEHLE](https://www.leuphana.de/institute/imd/personen/kathrin-padberg-gehle.html) for providing the Bickley jet example.

This project is funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - SFB 1615 - 503850735.

<p align="center">
  <img src="./Logos/tu_SMART_LOGO_02.jpg" height="105"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>
