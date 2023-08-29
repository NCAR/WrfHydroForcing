from gdrive_download import maybe_download_testdata
import subprocess


def run():
    maybe_download_testdata()

    cmd = "pytest"
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    run()