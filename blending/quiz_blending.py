import numpy as np
from numpy.linalg import inv

NUM_QUAL = 2749898
RMSE_ZEROS = 3.84258
MYS_ZERO = RMSE_ZEROS**(2) * NUM_QUAL
# regularization matrix with ridge regression 1.35 (3) was the best 0.85 (2)
REG = 1.3


# returns M x 1 array
def A_Ts (A_T):
    arr = []
    for row in A_T:
        value = 0.5 * (sum(row**2) + MYS_ZERO)
        arr.append(value)
    return np.array(arr)

# add regularization
def regularize (A, reg):
    A.flat[::A.shape[1]+1] += reg * NUM_QUAL

def getAlphas (A):
    A_T = np.transpose(A)
    A_TA = np.matmul(A_T, A)
    regularize(A_TA, REG)
    return np.dot(inv(A_TA), A_Ts(A_T))


def main():

    print("Start Loading Data")
    # load ratings
    # james_ratings = np.loadtxt("data/james/k_10_epoch_40_reg_0.001000bias.dta").astype(float)
    # kevin_ratings = np.loadtxt("data/kevin/rSVD_1-31%.dta").astype(float)
    # svdpp = np.loadtxt("data/frank/SVDPP_predictions.dta").astype(float)
    svdpp1 = np.loadtxt("data/frank/SVDPP_prediction_qual4-04.dta").astype(float)
    # james1 = np.loadtxt("data/james/k_50_epoch_20_reg_0.1_eta_0.01_ABOVE.dta").astype(float)
    # james2 = np.loadtxt("data/james/k_50_epoch_20_reg_0.075_eta_0.01BEST.dta").astype(float)
    james3 = np.loadtxt("data/james/k_50_ep_100_reg_0.065000_eta_0.0100003-81.dta").astype(float)
    james4 = np.loadtxt("data/james/svdpp_k_50_ep_30_reg_0.005000_reg2_0.015000_reg3_0.015000_eta_0.007000_eta2_0.007000_eta3_0.007000_decay_0.9000004-08.dta").astype(float)
    # kevin1 = np.loadtxt("data/kevin/out_quiz.dta").astype(float)
    # kevin2 = np.loadtxt("data/kevin/qual/out_qual_svd_1.92.dta").astype(float)

    print("Finished Loading Data")

    models = []
    # append ratings
    # models.append(james_ratings)
    # models.append(kevin_ratings)
    # models.append(svdpp)
    models.append(svdpp1)
    # models.append(james1)
    # models.append(james2)
    models.append(james3)
    models.append(james4)
    # models.append(kevin2)
    # models.append(kevin1)



    models = np.array(models)
    A = np.transpose(models)

    alphas = getAlphas(A)
    print("alphas: ")
    print(alphas)

    ratings = np.dot(A, np.transpose(alphas))
    ratings = np.clip(ratings, 1, 5)

    f = open("output/current.dta", "w+")

    for i in range(len(ratings)):
        f.write(str(ratings[i]) + '\n')
    f.close()



if __name__ == "__main__":
    main()
