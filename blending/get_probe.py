import numpy as np
NUM_TRAINING = 102416306 - 2749898
NUM_QUAL = 2749898


def main():
    print("Start loading all.idx")
    probe = []
    index = 0
    for line in open('../um/all.idx'):
        row = int(line)
        if row == 4:
            probe.append(index)
        index += 1
    print("Done loading all.idx")
    print("length of probe: " + str(len(probe)))
    print(" ")

    print("Start loading all.dta")
    probe_ratings = []
    index = 0
    for line in open('../um_data/all.dta'):
        if len(probe) == 0:
            break
        if index == probe[0]:
            rating = np.fromstring(line, dtype=int, sep=' ')[3]
            probe_ratings.append(rating)
            probe.pop(0)
        index += 1
    print("Done loading all.dta")

    print("length of probe: " + str(len(probe_ratings)))

    f = open("probe_ratings.dta", "w+")
    for i in range(len(probe_ratings)):
        f.write(str(probe_ratings[i]) + '\n')
    f.close()



if __name__ == "__main__":
    main()