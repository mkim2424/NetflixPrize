#include "include.h"
#include "Baseline.h"
#include "Model.h"

std::string header = "/home/ec2-user/cs156b/svd/";
// std::string header = "/Users/jameswei/Documents/Caltech/cs156b/svd_cpp/";

void run(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
    Eigen::ArrayXXi& qual) {
    print_parameters(p);
    Baseline model = Baseline(p, train, valid, qual, header);
    model.trainModel(false);
    std::cout << "FINAL ERROR: " << model.getValidError() << "\n" << std::endl;
}

int main(int argc, char *argv[]) {
    Eigen::ArrayXXi all = Eigen::ArrayXXi(TRAINING_SIZE, 4);
    fill_all(all, header);
    Eigen::ArrayXXi train = Eigen::ArrayXXi(TRAINING_SIZE - PROBE_SIZE, 4);
    fill_not_probe(train, header);
    Eigen::ArrayXXi valid = Eigen::ArrayXXi(PROBE_SIZE, 4);
    fill_probe(valid, header);
    Eigen::ArrayXXi qual  = Eigen::ArrayXXi(QUAL_SIZE, 3);
    fill_qual(qual, header);

    Parameter p = {SVD_t, 0, 5, {0.005, 0.005}, {0.01, 0.01}, 0.95};
    // run(p, train, valid, qual);
    print_parameters(p);
    Baseline model = Baseline(p, train, valid, qual, header);
    model.trainModel(false);
    model.predictProbe();

    std::cout << "\a";
    return 0;
}