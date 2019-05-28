#ifndef TIME_SVD_H
#define TIME_SVD_H

#include "Model.h"
#include "include.h"

class Time_SVD : public Model {
private:
    Eigen::ArrayXXd base_i_bin;
    Eigen::VectorXd mean_date;
    Eigen::VectorXd alpha_u;
    std::set<int>   **dates_rated;
    std::unordered_map<int, double> **base_u_t;
    std::unordered_map<int, int> **freq; // map btw date and number of ratings
    Eigen::ArrayXXd base_i_f;
    Eigen::ArrayXXd alpha_p_u;

    double get_dev(int usr_num, int date);
    int lower_f(int usr_num, int date);
    void preProcess();
    void SGD(int epoch);

public:
    Time_SVD(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
        Eigen::ArrayXXi& qual, std::string header);
    ~Time_SVD();
    double predictValue(int usr_num, int mv_num, int date);
};

#endif
