#ifndef BASELINE_H
#define BASELINE_H

#include "Model.h"
#include "include.h"

class Baseline : public Model {
private:
    void preProcess();
    void SGD(int epoch);

public:
    Baseline(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
        Eigen::ArrayXXi& qual, std::string header);
    double predictValue(int usr_num, int mv_num, int date);
};

#endif
