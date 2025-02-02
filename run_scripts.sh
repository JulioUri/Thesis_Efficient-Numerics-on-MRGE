#!/bin/bash

# Permitir que el script continúe incluso si hay errores
set +e

# Define the paths to your Python scripts
SCRIPT01_DIR="./06010102-QSCNT-CNVRG-R"
SCRIPT02_DIR="./06010103-QSCNT-CNVRG-S"
SCRIPT03_DIR="./06010104-QSCNT-PRCSN"
SCRIPT04_DIR="./06010105-QSCNT-C"

SCRIPT05_DIR="./06010202-VORTX-CNVRG-R"
SCRIPT06_DIR="./06010203-VORTX-CNVRG-S"
SCRIPT07_DIR="./06010204-VORTX-PRCSN"
SCRIPT08_DIR="./06010205-VORTX-C"

SCRIPT09_DIR="./06010302-OSCLT-CNVRG-R"
SCRIPT10_DIR="./06010303-OSCLT-CNVRG-S"
SCRIPT11_DIR="./06010304-OSCLT-CNVRG-R"
SCRIPT12_DIR="./06010305-OSCLT-CNVRG-S"
SCRIPT13_DIR="./06010306-OSCLT-PRCSN/06010306-OSCLT-PRCSN-NZROQ"
SCRIPT14_DIR="./06010306-OSCLT-PRCSN/06010306-OSCLT-PRCSN-ZEROQ"
SCRIPT15_DIR="./06010307-OSCLT-C/06010307-OSCLT-CONVR-C-NZROQ"
SCRIPT16_DIR="./06010307-OSCLT-C/06010307-OSCLT-CONVR-C-ZEROQ"

SCRIPT17_DIR="./06020102-DGYRE-CNVRG-R"
SCRIPT18_DIR="./06020103-DGYRE-CNVRG-S"
SCRIPT19_DIR="./06020104-DGYRE-CNVRG-R"
SCRIPT20_DIR="./06020105-DGYRE-CNVRG-S"
SCRIPT21_DIR="./06020106-DGYRE-PRCSN/06020106-DGYRE-PRCSN-NZROQ"
SCRIPT22_DIR="./06020106-DGYRE-PRCSN/06020106-DGYRE-PRCSN-ZEROQ"
SCRIPT23_DIR="./06020107-DGYRE-C/06020107-DGYRE-CONVR-C-NZROQ"
SCRIPT24_DIR="./06020107-DGYRE-C/06020107-DGYRE-CONVR-C-ZEROQ"

SCRIPT25_DIR="./06020202-BICKL-CNVRG-R"
SCRIPT26_DIR="./06020203-BICKL-CNVRG-S"
SCRIPT27_DIR="./06020204-BICKL-CNVRG-R"
SCRIPT28_DIR="./06020205-BICKL-CNVRG-S"
SCRIPT29_DIR="./06020206-BICKL-PRCSN/06020206-BICKL-PRCSN-NZROQ"
SCRIPT30_DIR="./06020206-BICKL-PRCSN/06020206-BICKL-PRCSN-ZEROQ"
SCRIPT31_DIR="./06020207-BICKL-C/06020207-BICKL-CONVR-C-NZROQ"
SCRIPT32_DIR="./06020207-BICKL-C/06020207-BICKL-CONVR-C-ZEROQ"

SCRIPT33_DIR="./06020302-DATA1-CNVRG-R"
SCRIPT34_DIR="./06020303-DATA1-CNVRG-S"
SCRIPT35_DIR="./06020304-DATA1-CNVRG-R"
SCRIPT36_DIR="./06020305-DATA1-CNVRG-S"
SCRIPT37_DIR="./06020306-DATA1-PRCSN/06020306-DATA1-PRCSN-NZROQ"
SCRIPT38_DIR="./06020306-DATA1-PRCSN/06020306-DATA1-PRCSN-ZEROQ"
SCRIPT39_DIR="./06020307-DATA1-C/06020307-DATA1-CONVR-C-NZROQ"
SCRIPT40_DIR="./06020307-DATA1-C/06020307-DATA1-CONVR-C-ZEROQ"


# Save the current directory
COMMON_DIR=$(pwd)


##################
# Quiescent Flow #
##################

# Run the first script
#cd "$SCRIPT01_DIR" || { echo "Failed to change directory to $SCRIPT01_DIR"; exit 1; }
#python3 a06_CONVRG_QSCNT.py || { echo "Failed to run FIRST script."; exit 1; }
#cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the second script
#cd "$SCRIPT02_DIR" || { echo "Failed to change directory to $SCRIPT02_DIR"; exit 1; }
#python3 a06_CONVRG_QSCNT.py || { echo "Failed to run SECOND script."; exit 1; }
#cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the third script
#cd "$SCRIPT03_DIR" || { echo "Failed to change directory to $SCRIPT03_DIR"; exit 1; }
#python3 a06_PRECSN_QSCNT.py || { echo "Failed to run THIRD script."; exit 1; }
#cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the fourth script
#cd "$SCRIPT04_DIR" || { echo "Failed to change directory to $SCRIPT04_DIR"; exit 1; }
#python3 a06_ERRORS_QSCNT.py || { echo "Failed to run FOURTH script."; exit 1; }
#cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }


###############
# Vortex Flow #
###############

# Run the fifth script
#cd "$SCRIPT05_DIR" || { echo "Failed to change directory to $SCRIPT05_DIR"; exit 1; }
#python3 a06_CONVRG_VORTX.py || { echo "Failed to run FIFTH script."; exit 1; }
#cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the sixth script
#cd "$SCRIPT06_DIR" || { echo "Failed to change directory to $SCRIPT06_DIR"; exit 1; }
#python3 a06_CONVRG_VORTX.py || { echo "Failed to run SIXTH script."; exit 1; }
#cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the seventh script
#cd "$SCRIPT07_DIR" || { echo "Failed to change directory to $SCRIPT07_DIR"; exit 1; }
#python3 a06_PRECSN_VORTX.py || { echo "Failed to run SEVENTH script."; exit 1; }
#cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the eigth script
#cd "$SCRIPT08_DIR" || { echo "Failed to change directory to $SCRIPT08_DIR"; exit 1; }
#python3 a06_ERRORS_VORTX.py || { echo "Failed to run EIGTH script."; exit 1; }
#cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }


##########################
# Oscillatory Background #
##########################

# Run the nineth script
cd "$SCRIPT09_DIR" || { echo "Failed to change directory to $SCRIPT09_DIR"; exit 1; }
python3 a06_CONVRG_OSCLT.py || { echo "Failed to run NINETH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the tenth script
cd "$SCRIPT10_DIR" || { echo "Failed to change directory to $SCRIPT10_DIR"; exit 1; }
python3 a06_CONVRG_OSCLT.py || { echo "Failed to run TENTH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the eleventh script
cd "$SCRIPT11_DIR" || { echo "Failed to change directory to $SCRIPT11_DIR"; exit 1; }
python3 a06_CONVRG_OSCLT.py || { echo "Failed to run ELEVENTH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the twelfth script
cd "$SCRIPT12_DIR" || { echo "Failed to change directory to $SCRIPT12_DIR"; exit 1; }
python3 a06_CONVRG_OSCLT.py || { echo "Failed to run TWELFTH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the thirdteenth script
cd "$SCRIPT13_DIR" || { echo "Failed to change directory to $SCRIPT13_DIR"; exit 1; }
python3 a06_PRECSN_OSCLT.py || { echo "Failed to run 13TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 14th script
cd "$SCRIPT14_DIR" || { echo "Failed to change directory to $SCRIPT14_DIR"; exit 1; }
python3 a06_PRECSN_OSCLT.py || { echo "Failed to run 14TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 15th script
cd "$SCRIPT15_DIR" || { echo "Failed to change directory to $SCRIPT15_DIR"; exit 1; }
python3 a06_ERRORS_OSCLT.py || { echo "Failed to run 15TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 16th script
cd "$SCRIPT16_DIR" || { echo "Failed to change directory to $SCRIPT16_DIR"; exit 1; }
python3 a06_ERRORS_OSCLT.py || { echo "Failed to run 16TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }


#####################
# Double Gyre Field #
#####################

# Run the 17th script
cd "$SCRIPT17_DIR" || { echo "Failed to change directory to $SCRIPT17_DIR"; exit 1; }
python3 a06_CONVRG_DGYRE.py || { echo "Failed to run 17TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 18th script
cd "$SCRIPT18_DIR" || { echo "Failed to change directory to $SCRIPT18_DIR"; exit 1; }
python3 a06_CONVRG_DGYRE.py || { echo "Failed to run 18TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 19th script
cd "$SCRIPT19_DIR" || { echo "Failed to change directory to $SCRIPT19_DIR"; exit 1; }
python3 a06_CONVRG_DGYRE.py || { echo "Failed to run 19TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 20th script
cd "$SCRIPT20_DIR" || { echo "Failed to change directory to $SCRIPT20_DIR"; exit 1; }
python3 a06_CONVRG_DGYRE.py || { echo "Failed to run 20TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 21st script
cd "$SCRIPT21_DIR" || { echo "Failed to change directory to $SCRIPT21_DIR"; exit 1; }
python3 a06_PRECSN_DGYRE.py || { echo "Failed to run 21st script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 22nd script
cd "$SCRIPT22_DIR" || { echo "Failed to change directory to $SCRIPT22_DIR"; exit 1; }
python3 a06_PRECSN_DGYRE.py || { echo "Failed to run 22nd script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 23rd script
cd "$SCRIPT23_DIR" || { echo "Failed to change directory to $SCRIPT23_DIR"; exit 1; }
python3 a06_ERRORS_DGYRE.py || { echo "Failed to run 23rd script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 24th script
cd "$SCRIPT24_DIR" || { echo "Failed to change directory to $SCRIPT24_DIR"; exit 1; }
python3 a06_ERRORS_DGYRE.py || { echo "Failed to run 24TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }


###############
# Bickley Jet #
###############

# Run the 25th script
cd "$SCRIPT25_DIR" || { echo "Failed to change directory to $SCRIPT25_DIR"; exit 1; }
python3 a06_CONVRG_BICKL.py || { echo "Failed to run 25TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 26th script
cd "$SCRIPT26_DIR" || { echo "Failed to change directory to $SCRIPT26_DIR"; exit 1; }
python3 a06_CONVRG_BICKL.py || { echo "Failed to run 26TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 27th script
cd "$SCRIPT27_DIR" || { echo "Failed to change directory to $SCRIPT27_DIR"; exit 1; }
python3 a06_CONVRG_BICKL.py || { echo "Failed to run 27TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 28th script
cd "$SCRIPT28_DIR" || { echo "Failed to change directory to $SCRIPT28_DIR"; exit 1; }
python3 a06_CONVRG_BICKL.py || { echo "Failed to run 28TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 29th script
cd "$SCRIPT29_DIR" || { echo "Failed to change directory to $SCRIPT29_DIR"; exit 1; }
python3 a06_PRECSN_BICKL.py || { echo "Failed to run 29TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 30th script
cd "$SCRIPT30_DIR" || { echo "Failed to change directory to $SCRIPT30_DIR"; exit 1; }
python3 a06_PRECSN_BICKL.py || { echo "Failed to run 30th script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 31st script
cd "$SCRIPT31_DIR" || { echo "Failed to change directory to $SCRIPT31_DIR"; exit 1; }
python3 a06_ERRORS_BICKL.py || { echo "Failed to run 31st script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 32nd script
cd "$SCRIPT32_DIR" || { echo "Failed to change directory to $SCRIPT32_DIR"; exit 1; }
python3 a06_ERRORS_BICKL.py || { echo "Failed to run 32nd script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }


################
# Faraday flow #
################

# Run the 33rd script
cd "$SCRIPT33_DIR" || { echo "Failed to change directory to $SCRIPT33_DIR"; exit 1; }
python3 a06_CONVRG_DATA1.py || { echo "Failed to run 33rd script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 34th script
cd "$SCRIPT34_DIR" || { echo "Failed to change directory to $SCRIPT34_DIR"; exit 1; }
python3 a06_CONVRG_DATA1.py || { echo "Failed to run 34TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 35th script
cd "$SCRIPT35_DIR" || { echo "Failed to change directory to $SCRIPT35_DIR"; exit 1; }
python3 a06_CONVRG_DATA1.py || { echo "Failed to run 35TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 36th script
cd "$SCRIPT36_DIR" || { echo "Failed to change directory to $SCRIPT36_DIR"; exit 1; }
python3 a06_CONVRG_DATA1.py || { echo "Failed to run 36TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 37th script
cd "$SCRIPT37_DIR" || { echo "Failed to change directory to $SCRIPT37_DIR"; exit 1; }
python3 a06_PRECSN_DATA1.py || { echo "Failed to run 37TH script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 38th script
cd "$SCRIPT38_DIR" || { echo "Failed to change directory to $SCRIPT38_DIR"; exit 1; }
python3 a06_PRECSN_DATA1.py || { echo "Failed to run 38th script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 39th script
cd "$SCRIPT39_DIR" || { echo "Failed to change directory to $SCRIPT39_DIR"; exit 1; }
python3 a06_ERRORS_DATA1.py || { echo "Failed to run 39th script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

# Run the 40th script
cd "$SCRIPT40_DIR" || { echo "Failed to change directory to $SCRIPT40_DIR"; exit 1; }
python3 a06_ERRORS_DATA1.py || { echo "Failed to run 40th script."; exit 1; }
cd "$COMMON_DIR" || { echo "Failed to return to common directory"; exit 1; }

echo "All scripts have been executed."
