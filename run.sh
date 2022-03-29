NTHREADS="1"
PATH="/home/ssane/Downloads/vtiData/"
FILENAME1="f1Ensemble"
FILENAME2="f2Ensemble"
NFILES="40"
NBINS="22"

#time ./uniform_fs $PATH $FILENAME1 $FILENAME2 $NFILES $NBINS $NTHREADS
time ./gaussian_fs $PATH $FILENAME1 $FILENAME2 $NFILES $NTHREADS

