import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

NUM_TRAINING = 102416306 - 2749898
NUM_QUAL = 2749898



def main():
    # predictions on probe
    probe_predictions = [#"data/kevin/probe/out_probe_svd_1.92.dta",
                         # "data/misc/all_threes_probe.dta",
                         "data/frank/SVDPP_prediction_probe4-04.dta"]                                    
    # predictions on quiz
    qual_predictions = [#"data/kevin/qual/out_qual_svd_1.92.dta",
                        # "data/misc/all_threes_qual.dta",
                        "data/frank/SVDPP_prediction_qual4-04.dta"]
    num_models = 0


    # check if length of probe_predictions matches length of quiz predictions
    if len(probe_predictions) == len(qual_predictions):
        num_models = len(probe_predictions)
        print("GOOD TO GO")
    else:
        print("WE HAVE A PROBLEM")


    # load predictions on probe
    X_train = []
    for preds in probe_predictions:
        X_train.append(np.loadtxt(preds).astype(float))
    # load probe (just the ratings)
    probe = np.loadtxt('probe_ratings.dta').astype(int)

    # load predictions on quiz
    X_test = []
    for preds in qual_predictions:
        X_test.append(np.loadtxt(preds).astype(float))


    # Transpose the matrices so they have dimension NUM_QUAL/NUM_PROBE x num_models
    X_train = np.array(X_train).T
    print(X_train.shape)
    X_test = np.array(X_test).T
    print(X_test.shape)

    print("starting training")
    # verbosity = 2, eta = 0.1, n_estimators = 100
    model = XGBRegressor()
    model.fit(X_train, probe)
    print("finish training")

    # get training error
    ratings_probe = model.predict(X_train)
    print("Training error: ", mean_squared_error(probe, ratings_probe))

    # get qual ratings
    ratings_qual = model.predict(X_test)
    ratings_qual = np.clip(ratings_qual, 1, 5)
    

    f = open("output/frank_super.dta", "w+")

    for i in range(len(ratings_qual)):
        f.write(str(ratings_qual[i]) + '\n')
    f.close()

if __name__ == "__main__":
    main()