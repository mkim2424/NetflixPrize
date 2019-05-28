#ifndef UTIL_H
#define UTIL_H

#include <set>
#include <iostream>
#include <ctime>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include "values.h"

enum ModelType { Baseline_t, Time_baseline_t, Time_SVD_t, SVD_t, SVDpp_t, 
	Time_SVDpp_t, FactorNM_t };

struct Parameter {
    ModelType mt;
    int k;
    int maxEpoch;
    std::vector<double> reg;
    std::vector<double> eta;
    double decay;
};

void print(std::string str);
void println(std::string str);
void print(std::string str, int val);
void print(std::string str, double val);
void print_parameters(Parameter p);

/*
 * Used for clipping values between 1 and 5.
 *
 * @param value the input value to be clipped.
 */
double clip(double value);

// Given a date, return its bin, 0 indexed to 30. Each bin is 10 weeks.
int get_bin(int date);

double sign(double val);

#endif
