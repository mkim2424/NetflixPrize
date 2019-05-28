#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <cassert>
#include <Eigen/Dense>

#include "util.h"
#include "values.h"

// std::string header = "/Users/jameswei/Documents/Caltech/cs156b/svd/";
std::string header = "/home/ec2-user/cs156b/svd/";

int main(int argc, char *argv[]) {
    time_t before;
    int usr_num, mv_num, date, rate, ind;
    int count = 0, training_count = 0, not_probe_count = 0, probe_count = 0;

    Eigen::ArrayXXi train(TRAINING_SIZE, 4);
    Eigen::ArrayXXi probe(PROBE_SIZE, 4);
    Eigen::ArrayXXi not_probe(TRAINING_SIZE - PROBE_SIZE, 4);
    Eigen::ArrayXXi qual(QUAL_SIZE, 3);

    before = time(nullptr);
    std::ifstream fin_data(header + "data/all.dta");
    std::ifstream fin_idx(header + "data/all.idx");
    assert(fin_data.is_open() && fin_idx.is_open());
    std::cout << "Loading data..." << std::endl;

    while (fin_data >> usr_num >> mv_num >> date >> rate) {
        fin_idx >> ind;

        // Not probe
        if (ind < 4) {
            not_probe(not_probe_count, 0) = usr_num;
            not_probe(not_probe_count, 1) = mv_num;
            not_probe(not_probe_count, 2) = date;
            not_probe(not_probe_count, 3) = rate;
            not_probe_count++;
        }

        // Probe
        if (ind == 4) {
            probe(probe_count, 0) = usr_num;
            probe(probe_count, 1) = mv_num;
            probe(probe_count, 2) = date;
            probe(probe_count, 3) = rate;
            probe_count++;
        }

        // Not qual
        if (ind != 5) {
            train(training_count, 0) = usr_num;
            train(training_count, 1) = mv_num;
            train(training_count, 2) = date;
            train(training_count, 3) = rate;
            training_count++;
        }

        if (++count % 10000000 == 0) {
            std::cout << "." << std::flush;
        }
    }

    assert(training_count == TRAINING_SIZE);
    assert(probe_count == PROBE_SIZE);
    std::cout << "\nDone loading data. Took " << time(nullptr) - before <<
        "s.\n" << std::endl;

    std::string folder = "data/";

    std::ofstream fout_all_train(header + folder + "all_train.dta");
    assert(fout_all_train.is_open());
    std::cout << "Writing to all_train..." << std::flush;
    before = time(nullptr);
    for (int i = 0; i < TRAINING_SIZE; i++) {
        fout_all_train << train(i, 0) << " " << train(i, 1) << " " 
            << train(i, 2) << " " << train(i, 3) << "\n";
    }
    std::cout << "done. Took " << time(nullptr) - before << "s.\n" << std::endl;

    std::ofstream fout_valid(header + folder + "valid.dta");
    assert(fout_valid.is_open());
    // int *shuffle = shuffleIndicies(TRAINING_SIZE);

    std::ofstream fout_not_probe(header + folder + "not_probe.dta");
    assert(fout_not_probe.is_open());
    std::cout << "Writing to not_probe..." << std::flush;
    before = time(nullptr);
    for (int i = 0; i < not_probe.rows(); i++) {
        fout_not_probe << not_probe(i, 0) << " " << not_probe(i, 1) << " " << not_probe(i, 2) 
            << " " << not_probe(i, 3) << "\n";
    }
    std::cout << "done. Took " << time(nullptr) - before << "s.\n" << std::endl;
    
    std::ofstream fout_probe(header + folder + "probe.dta");
    assert(fout_probe.is_open());
    std::cout << "Writing to probe..." << std::flush;
    for (int i = 0; i < probe.rows(); i++) {
        fout_probe << probe(i, 0) << " " << probe(i, 1) << " " << probe(i, 2) 
            << " " << probe(i, 3) << "\n";
    }
    std::cout << "done.\n" << std::endl;

    return 0;
}
