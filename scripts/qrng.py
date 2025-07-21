# With `cmake -DENABLE_RNDFILE=ON -DENABLE_RDRAND=OFF -DENABLE_DEVRAND=OFF ..`,
# Qrack can read random number generation data in from a set of files, like
# from https://github.com/sbalian/quantum-random

from pathlib import Path
import qrandom
import sys


def main():
    pages = 1
    if len(sys.argv) > 1:
        pages = int(sys.argv[1])

    qrng = qrandom.QuantumRandom()
    with open(Path.home() / ".qrack" / "rng" / "qrng.bin", "wb") as file:
        for i in range(pages):
            for _ in range(1024):
                file.write(
                    (qrng._get_rand_int64()).to_bytes(8, byteorder="big", signed=False)
                )


if __name__ == "__main__":
    sys.exit(main())
