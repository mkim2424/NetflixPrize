#include "include.h"
#include "Time_SVD.h"

std::string header = "/home/ec2-user/cs156b/svd/";
// std::string header = "/Users/jameswei/Documents/Caltech/cs156b/svd/";

int main(int argc, char *argv[]) {
    // Eigen::ArrayXXi train = Eigen::ArrayXXi(TRAINING_SIZE, 4);
    // fill_all(train, header);
    Eigen::ArrayXXi train = Eigen::ArrayXXi(TRAINING_SIZE - PROBE_SIZE, 4);
    fill_not_probe(train, header);
    Eigen::ArrayXXi valid = Eigen::ArrayXXi(PROBE_SIZE, 4);
    fill_probe(valid, header);
    Eigen::ArrayXXi qual  = Eigen::ArrayXXi(QUAL_SIZE, 3);
    fill_qual(qual, header);

    Parameter p = { Time_SVD_t, 200, 20, 
        { 0.005, 0.005, 0.015, 0.005, 50,   0.005, 0, 0, 1.1e-8, 70   }, 
        { 0.01,  0.01,  0.01,  0.01,  1e-5, 0.01,  0, 0, 0.01,   1e-5 }, 0.95 };
    print_parameters(p);
    Time_SVD model = Time_SVD(p, train, valid, qual, header);
    model.trainModel(false);
    model.predictProbe();

    std::cout << "\a";
    return 0;
}
