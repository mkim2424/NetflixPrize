#ifndef SVD_H
#define SVD_H

#include "Model.h"
#include "include.h"

class SVD : public Model {
private:
    void preProcess();
    void SGD(int epoch);

public:
    SVD(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
        Eigen::ArrayXXi& qual, std::string header);
    double predictValue(int usr_num, int mv_num, int date);
};

#endif
