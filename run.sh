NTHREADS="1"
PATH="./vtiData/"
FILENAME1="f1Ensemble"
FILENAME2="f2Ensemble"
NFILES="40"
NBINS="22"

time ./build/uniform_fs $PATH $FILENAME1 $FILENAME2 $NFILES $NBINS $NTHREADS #> cpu.out
time ./build/uniform_gpu $PATH $FILENAME1 $FILENAME2 $NFILES $NBINS $NTHREADS #> gpu.out
#time ./gaussian_fs $PATH $FILENAME1 $FILENAME2 $NFILES $NTHREADS

