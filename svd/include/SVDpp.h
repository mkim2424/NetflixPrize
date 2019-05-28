#ifndef SVDPP_H
#define SVDPP_H

#include "Model.h"
#include "include.h"

class SVDpp : public Model {
private:
    Eigen::VectorXi  loc;
    Eigen::ArrayXXd  Y;
    Eigen::ArrayXXd  sum_Y;
    std::vector<int> **movies_watched;

    int get_num_movies_with_rating(int usr_num);
    void update_sum_Y(int usr_num);
    void preProcess();
    void SGD(int epoch);

public:
    SVDpp(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
        Eigen::ArrayXXi& qual, std::string header);
    double predictValue(int usr_num, int mv_num, int date);
};

#endif
