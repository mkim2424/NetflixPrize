#ifndef TIME_SVDPP_H
#define TIME_SVDPP_H

#include "Model.h"
#include "include.h"

class Time_SVDpp : public Model {
private:
    Eigen::ArrayXXd base_i_bin;
    Eigen::VectorXd mean_date;
    Eigen::VectorXd alpha_u;
    std::set<int>   **dates_rated;
    std::unordered_map<int, double> **base_u_t;
    std::unordered_map<int, int> **freq; // map btw date and number of ratings
    Eigen::ArrayXXd base_i_f;
    Eigen::ArrayXXd alpha_p_u;

    Eigen::VectorXi loc;
    Eigen::ArrayXXd Y;
    Eigen::ArrayXXd sum_Y;
    std::vector<int> **movies_watched;

    double get_dev(int usr_num, int date);
    int lower_f(int usr_num, int date);
    void update_sum_Y(int usr_num);
    int get_num_movies_with_rating(int usr_num);
    void preProcess();
    void SGD(int epoch);

public:
    Time_SVDpp(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
        Eigen::ArrayXXi& qual, std::string header);
    ~Time_SVDpp();
    double predictValue(int usr_num, int mv_num, int date);
};

#endif
