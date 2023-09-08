from gdrive_download import maybe_download_testdata
import subprocess


def run():
    maybe_download_testdata()

    cmd = "mpirun -n 4 python -m pytest --with-mpi"
    test_proc = subprocess.run(cmd, shell=True)
    exit(test_proc.returncode)


if __name__ == "__main__":
    run()
