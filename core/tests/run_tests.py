from gdrive_download import maybe_download_testdata
import subprocess

COLOR_GREEN = "\033[92m"
COLOR_NORM = "\033[0m"
COLOR_RED = "\033[91m"


def run():
    maybe_download_testdata()

    failed = 0

    print ("Testing without MPI...")

    cmd = "python -m pytest --with-mpi"
    test_proc = subprocess.run(cmd, shell=True)
    if test_proc.returncode != 0:
        print("%s----Tests not using MPI failed!!!%s" % (COLOR_RED, COLOR_NORM))
        failed += 1
    else:
        print("%s---- Tests not using MPI passed%s" % (COLOR_GREEN, COLOR_NORM))

    print ("Testing with 1 MPI process...")

    cmd = "mpirun -n 1 python -m pytest --with-mpi"
    test_proc = subprocess.run(cmd, shell=True)
    if test_proc.returncode != 0:
        print("%s---- MPI nprocs = 1 tests failed!!!%s" % (COLOR_RED, COLOR_NORM))
        failed += 1
    else:
        print("%s---- MPI nprocs = 1 tests passed%s" % (COLOR_GREEN, COLOR_NORM))

    print ("Testing with 4 MPI processes...")

    cmd = "mpirun -n 4 python -m pytest --with-mpi"
    test_proc = subprocess.run(cmd, shell=True)
    if test_proc.returncode != 0:
        print("%s---- MPI nprocs = 4 tests failed!!!%s" % (COLOR_RED, COLOR_NORM))
        failed += 1
    else:
        print("%s---- MPI nprocs = 4 tests passed%s" % (COLOR_GREEN, COLOR_NORM))

    
    print ("Testing with 8 MPI processes...")

    cmd = "mpirun -n 8 python -m pytest --with-mpi"
    test_proc = subprocess.run(cmd, shell=True)
    if test_proc.returncode != 0:
        print("%s---- MPI nprocs = 8 tests failed!!!%s" % (COLOR_RED, COLOR_NORM))
        failed += 1
    else:
        print("%s---- MPI nprocs = 8 tests passed%s" % (COLOR_GREEN, COLOR_NORM))

    if failed > 0:
        print("%s---- Total %s MPI configurations failed!!!%s" % (failed, COLOR_RED, COLOR_NORM))
        exit(-1)
        
    print("%s******** All tests passed ***********%s" % (COLOR_GREEN, COLOR_NORM))
    exit(0)
    



if __name__ == "__main__":
    run()
