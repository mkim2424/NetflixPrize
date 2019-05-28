#include "Model.h"

Model::Model(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
    Eigen::ArrayXXi& qual, std::string header) {
    this->p        = p;
    this->k        = p.k;
    this->maxEpoch = p.maxEpoch;
    this->reg      = p.reg;
    this->eta      = p.eta;
    this->decay    = p.decay;
    this->decay_factor = 1;
    this->train    = train;
    this->valid    = valid;
    this->qual     = qual;
    this->header   = header;

    this->base_u      = Eigen::VectorXd::Zero(M);
    this->base_i      = Eigen::VectorXd::Zero(N);
    this->U           = Eigen::ArrayXXd::Random(M, k) / 100;
    this->V           = Eigen::ArrayXXd::Random(N, k) / 100;
}

double Model::getValidError() {
    double error = 0;

    for (int i = 0; i < valid.rows(); i++) {
        Eigen::VectorXi row = valid.row(i);
        int usr_num = row[0], mv_num = row[1], date = row[2], rate = row[3];
        double err = rate - predictValue(usr_num, mv_num, date);
        error += pow(err, 2);
    }

    return sqrt(error / valid.rows());
}

bool Model::trainModel(bool earlyStop) {
    std::cout << "TRAINING MODEL WITH k = " << k << std::endl;
    time_t before;
    double prevError = getValidError(), currError;

    std::cout << "GOT FIRST ERROR" << std::endl;

    for (int i = 0; i < maxEpoch; i++) {
        before = time(0);
        std::cout << "EPOCH: " << i + 1 << "/" << maxEpoch << std::flush;
        SGD(i);
        currError = getValidError();
        std::cout << ". ERROR: " << currError << " (" <<
            time(0) - before << "s)\n" << std::endl;

        if (earlyStop && currError > prevError) {
            std::cout << "EARLY STOPPING" << std::endl;
            break;
        }

        prevError = currError;
        decay_factor *= decay;
    }

    std::cout << "DONE TRAINING DATA.\n" << std::endl;
    return true;
}

void Model::predictQual() {
    std::cout << "PREDICTING QUAL VALUES..." << std::flush;
    std::ofstream fout_qual(header + "results/qual.dta");
    assert(fout_qual.is_open());

    for (int i = 0; i < qual.rows(); i++) {
        Eigen::VectorXi row = qual.row(i);
        int usr_num = row[0], mv_num = row[1], date = row[2];
        fout_qual << clip(predictValue(usr_num, mv_num, date)) << "\n";
    }

    std::cout << "DONE." << std::endl;
}

void Model::predictProbe() {
    std::cout << "PREDICTING PROBE VALUES..." << std::flush;
    std::ofstream fout_probe(header + "results/probe.dta");
    assert(fout_probe.is_open());

    for (int i = 0; i < valid.rows(); i++) {
        Eigen::VectorXi row = valid.row(i);
        int usr_num = row[0], mv_num = row[1], date = row[2];
        fout_probe << clip(predictValue(usr_num, mv_num, date)) << "\n";
    }

    std::cout << "DONE." << std::endl;
}