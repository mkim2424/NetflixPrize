#ifndef MODEL_H
#define MODEL_H

#include "util.h"
#include "include.h"

class Model {
protected:
    Parameter p;
    int k, maxEpoch;
    std::vector<double> reg;
    std::vector<double> eta;
    double decay, decay_factor;

    Eigen::ArrayXXi train;              // input matrix with ratings
    Eigen::ArrayXXi valid;
    Eigen::ArrayXXi qual;
    std::string     header;
    
    Eigen::VectorXd base_u;             // baseline for useres
    Eigen::VectorXd base_i;             // baseline for items
    Eigen::ArrayXXd U;
    Eigen::ArrayXXd V;

    /*
     * Performs the necessary preprocessing such as initializing
     * arrays to random values.
     */
    virtual void preProcess() = 0;

    /*
     * Runs one epoch of stochastic gradient descent, looping once 
     * through the training data in a random order. Updates U, V,
     * baselines and the implicit factors matrix.
     */
    virtual void SGD(int epoch) = 0;

public:
    Model(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
        Eigen::ArrayXXi& qual, std::string header);

    // Used to train the model. Calls helper function SGD <maxEpoch> times.
    // Implementss early stopping
    bool trainModel(bool earlyStop);

    /*
     * Predict a rating given the user number, movie number, and date with
     * the user and movie number being 0-indexed.
     */
    virtual double predictValue(int usr_num, int mv_num, int date) = 0;

    // Computes the RMSE error on the probe data.
    double getValidError();

    void predictQual();
    void predictProbe();
};

#endif
