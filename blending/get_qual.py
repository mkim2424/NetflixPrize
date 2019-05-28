import numpy as np
NUM_TRAINING = 102416306 - 2749898
NUM_QUAL = 2749898


def main():
    print("Start Loading Data")
    #max_rows = 102416306
    # all_data = np.loadtxt('../../um_data/all.dta', max_rows = 10000, usecols = (0,1,3)).astype(int)

    row_qual = 0
    qual = np.empty([NUM_QUAL, 3], dtype = int)
    for line in open('../../um_data/all.dta'):
        row = np.fromstring(line, dtype=int, sep=' ')[[0,1,3]]
        if row[2] == 0:
            qual[row_qual] = row[3]
            row_qual += 1
        

    print("Finished Loading Data")



    # training = []
    # qual = []
    # for row in all_data:
    #     if row[2] == 0:
    #         qual.append(row)
    #     else:
    #         training.append(row)
    #
    # training = np.array(training)
    # qual = np.array(qual)

    M = max(max(training[:,0]), max(qual[:,0])).astype(int) # users
    N = max(max(training[:,1]), max(qual[:,1])).astype(int) # movies
    print("Factorizing with ", M, " users, ", N, " movies.")

    reg = 0.007
    eta = 0.001 # learning rate
    K = 200
    epochs = 10

    print("Starting Training")
    U,V = train_model(M, N, K, eta, reg, training, max_epochs = epochs);
    ratings = get_ratings(U, V, qual)
    ratings = np.clip(ratings, 1, 5)

    f = open("result.dta", "w+")

    for i in range(len(ratings)):
        f.write(str(ratings[i]) + '\n')
    f.close()




if __name__ == "__main__":
    main()