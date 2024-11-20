#!/bin/bash
#
#$ -cwd
#$ -N z
#$ -l h=(comp1100|comp1101|comp1102|comp1103|comp1105|comp1107|comp1108|comp1109|comp1110|comp1111|comp1112|comp1113|comp1114|comp1115|comp1116|comp1117|comp1118|comp1119)
#$ -pe smp 1
#$ -l s_rt=70:00:00
#

source /opt/intel/oneapi/setvars.sh
export PATH=$PATH:/u/ajmk/chem1721/GroupPrograms/Molpro2022/molpro/ 
export OMP_NUM_THREADS=1
export MOLCAS=/u/ajmk/newc6739/Programs/OpenMolcas/OpenMolcas_2023_intel.mkl/
export SH=/u/ajmk/ptch0507/sh/src/oop/
export PYTHONPATH=$PYTHONPATH:$SH
export PYTHONPATH=$PYTHONPATH:$SH/testing/
export SHARC=/u/ajmk/newc6739/Programs/SHARC_3.0_implement_MASH/bin/

export ORIG=`pwd`
export SCR=$TMPDIR
echo "Node = " `cat /etc/hostname`
echo "Scratch = " $TMPDIR
echo "Start time = " `date`

function timesup {
  date
  cp -r $TMPDIR/* $ORIG
}

trap timesup SIGUSR1

cp -r $ORIG/* $SCR
cd $SCR
rm z.*
#run program
python3 $SH/main.py ethylene.json
cp -r * $ORIG
cd $ORIG
echo "End time = " `date`
