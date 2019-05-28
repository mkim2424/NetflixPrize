#ifndef FACTOR_NM_H
#define FACTOR_NM_H

#include "Model.h"
#include "include.h"

class FactorNM : public Model {
private:
	Eigen::VectorXi  loc;
	Eigen::ArrayXXd  X;
	Eigen::ArrayXXd  Y;
	Eigen::ArrayXXd  sum_X;
	Eigen::ArrayXXd  sum_Y;
	std::vector<int> **R_u;
	std::vector<int> **N_u;
	std::unordered_map<int, int> **ratings;

	// int get_num_movies_with_rating(int usr_num);
	void update_sum_X(int usr_num);
	void update_sum_Y(int usr_num);
    void preProcess();
    void SGD(int epoch);

public:
    FactorNM(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
        Eigen::ArrayXXi& qual, std::string header);
    double predictBias(int usr_num, int mv_num);
    double predictValue(int usr_num, int mv_num, int date);
};

#endif
