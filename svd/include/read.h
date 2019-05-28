#ifndef READ_H
#define READ_H

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "values.h"

void fill_all(Eigen::ArrayXXi& all, std::string header);
void fill_not_probe(Eigen::ArrayXXi& not_probe, std::string header);
void fill_probe(Eigen::ArrayXXi& probe, std::string header);
void fill_qual(Eigen::ArrayXXi& qual, std::string header);

#endif
