#include "SVD.h"

SVD::SVD(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
    Eigen::ArrayXXi& qual, std::string header) : 
    Model(p, train, valid, qual, header) {}

void SVD::preProcess() {}

void SVD::SGD(int epoch) {
    #pragma omp parallel for
    for (int i = 0; i < train.rows(); i++) {
        Eigen::VectorXi row = train.row(i);
        int usr_num = row[0], mv_num = row[1], date = row[2], rate = row[3];
        double err = rate - predictValue(usr_num, mv_num, date);
        Eigen::VectorXd old_p_u = U.row(usr_num);
        Eigen::VectorXd old_q_i = V.row(mv_num);

        base_u[usr_num] += eta[0] * decay_factor * (err - reg[0] * base_u[usr_num]);
        base_i[mv_num]  += eta[1] * decay_factor * (err - reg[1] * base_i[mv_num]);
        Eigen::VectorXd new_p_u = 
            old_p_u + eta[2] * decay_factor * (err * old_q_i - reg[2] * old_p_u);
        Eigen::VectorXd new_q_i = 
            old_q_i + eta[2] * decay_factor * (err * old_p_u - reg[2] * old_q_i);
        U.row(usr_num) = new_p_u;
        V.row(mv_num)  = new_q_i;
    }
}

double SVD::predictValue(int usr_num, int mv_num, int date) {
    Eigen::VectorXd p_u = U.row(usr_num);
    Eigen::VectorXd q_i = V.row(mv_num);
    return AVG_RATING + base_u[usr_num] + base_i[mv_num] + p_u.dot(q_i);
}
