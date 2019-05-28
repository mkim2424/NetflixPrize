import numpy as np
NUM_QUAL = 2749898
NUM_PROBE = 1374739

def main():
    f = open("data/misc/all_mean_probe.dta", "w+")
    for i in range(NUM_PROBE):
        f.write(str(3.7) + '\n')
    f.close()

    f = open("data/misc/all_mean_qual.dta", "w+")
    for i in range(NUM_QUAL):
        f.write(str(3.7) + '\n')
    f.close()


if __name__ == "__main__":
    main()
