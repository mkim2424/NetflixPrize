#include "Baseline.h"

Baseline::Baseline(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
    Eigen::ArrayXXi& qual, std::string header) : 
    Model(p, train, valid, qual, header) {}

void Baseline::preProcess() {}

void Baseline::SGD(int epoch) {
    #pragma omp parallel for
    for (int i = 0; i < train.rows(); i++) {
        Eigen::VectorXi row = train.row(i);
        int usr_num = row[0], mv_num = row[1], date = row[2], rate = row[3];
        double err = rate - predictValue(usr_num, mv_num, date);

        base_u[usr_num] += 
            eta[0] * decay_factor * (err - reg[0] * base_u[usr_num]);
        base_i[mv_num]  += 
            eta[1] * decay_factor * (err - reg[1] * base_i[mv_num]);
    }
}

double Baseline::predictValue(int usr_num, int mv_num, int date) {
    return AVG_RATING + base_u[usr_num] + base_i[mv_num];
}
