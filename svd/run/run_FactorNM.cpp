#include "include.h"
#include "FactorNM.h"
#include "Model.h"

std::string header = "/home/ec2-user/cs156b/svd/";
// std::string header = "/Users/jameswei/Documents/Caltech/cs156b/svd_cpp/";

int main(int argc, char *argv[]) {
    // Eigen::ArrayXXi train = Eigen::ArrayXXi(TRAINING_SIZE, 4);
    // fill_all(train, header);
    Eigen::ArrayXXi train = Eigen::ArrayXXi(TRAINING_SIZE - PROBE_SIZE, 4);
    fill_not_probe(train, header);
    Eigen::ArrayXXi valid = Eigen::ArrayXXi(PROBE_SIZE, 4);
    fill_probe(valid, header);
    Eigen::ArrayXXi qual  = Eigen::ArrayXXi(QUAL_SIZE, 3);
    fill_qual(qual, header);

    Parameter p = {FactorNM_t, 200, 7, {.04, .04, .04}, {.002, .002, .002}, 0.9};
    print_parameters(p);
    FactorNM model = FactorNM(p, train, valid, qual, header);
    model.trainModel(false);
    model.predictProbe();

    std::cout << "\a";
    return 0;
}