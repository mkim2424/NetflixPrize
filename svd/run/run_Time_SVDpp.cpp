#include "include.h"
#include "Time_SVDpp.h"

std::string header = "/home/ec2-user/cs156b/svd/";
// std::string header = "/Users/jameswei/Documents/Caltech/cs156b/svd_cpp/";

void run(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
    Eigen::ArrayXXi& qual) {
    print_parameters(p);
    Time_SVDpp model = Time_SVDpp(p, train, valid, qual, header);
    model.trainModel(true);
    std::cout << "FINAL ERROR: " << model.getValidError() << "\n" << std::endl;
}

int main(int argc, char *argv[]) {
    // Eigen::ArrayXXi train = Eigen::ArrayXXi(TRAINING_SIZE, 4);
    // fill_all(train, header);
    Eigen::ArrayXXi train = Eigen::ArrayXXi(TRAINING_SIZE - PROBE_SIZE, 4);
    fill_not_probe(train, header);
    Eigen::ArrayXXi valid = Eigen::ArrayXXi(PROBE_SIZE, 4);
    fill_probe(valid, header);
    Eigen::ArrayXXi qual  = Eigen::ArrayXXi(QUAL_SIZE, 3);
    fill_qual(qual, header);

    // Baseline (my parameters)
    Parameter p = { Time_SVDpp_t, 200, 20, 
        { 0.005, 0.005, 0.015, 0.005, 50,   0.005, 0, 0, 1.1e-8, 50   }, 
        { 0.01,  0.01,  0.01,  0.01,  1e-5, 0.01,  0, 0, 0.01,   1e-5 }, 0.95 };
    // run(p, train, valid, qual);
    print_parameters(p);
    Time_SVDpp model = Time_SVDpp(p, train, valid, qual, header);
    model.trainModel(false);
    model.predictProbe();

    // Paper's parameters
    // Parameter p = { Time_SVDpp_t, 50, 20, 
    //     { 2.6e-3,  2.6e-3,  0.0015,  9.3e-2,  400e-2, .231e-2,  0, 0, 1.1e-8,   70 },
    //     { 2.6e-3, .5e-3,     0.008, .115e-3,  3.11e-6, 2.7e-3,  0, 0, 2.36e-3, 1e-5 }, 0.90 };
    // print_parameters(p);
    // Time_SVDpp model = Time_SVDpp(p, train, valid, qual, header);
    // model.trainModel(false);
    // model.predictProbe();

    // Paper's parameters ROUND 2
    // Parameter p = { Time_SVDpp_t, 50, 20, 
    //     { 3e-3,   2.6e-3,  0.003,   9.3e-2,  400e-2, .231e-2,  0, 0, 0.8e-8,   70 },
    //     { 2.6e-3, .5e-3,     0.008, .115e-3,  3.11e-6, 2.7e-3,  0, 0, 2.36e-3, 1e-5 }, 0.90 };
    // run(p, train, valid, qual);

    std::cout << "\a";
    return 0;
}
