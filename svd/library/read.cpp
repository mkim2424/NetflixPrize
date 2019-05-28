#include "read.h"

void fill_all(Eigen::ArrayXXi& all, std::string header) {
    std::cout << "Loading all data..." << std::endl;
    std::ifstream fin_all(header + "data/all_train.dta");
    assert(fin_all.is_open());
    assert(all.rows() == TRAINING_SIZE);
    int usr_num, mv_num, date, rate;

    for (int i = 0; i < TRAINING_SIZE; i++) {
        fin_all >> usr_num >> mv_num >> date >> rate;
        all(i, 0) = usr_num - 1;
        all(i, 1) = mv_num - 1;
        all(i, 2) = date;
        all(i, 3) = rate;

        if ((i + 1) % 10000000 == 0) {
            std::cout << "." << std::flush;
        }
    }

    std::cout << ".\ndone.\n" << std::endl;
    fin_all.close();
}

void fill_not_probe(Eigen::ArrayXXi& not_probe, std::string header) {
    std::cout << "Loading not_probe data..." << std::endl;
    std::ifstream fin_not_probe(header + "data/not_probe.dta");
    assert(fin_not_probe.is_open());
    assert(not_probe.rows() == TRAINING_SIZE - PROBE_SIZE);
    int usr_num, mv_num, date, rate;

    for (int i = 0; i < TRAINING_SIZE - PROBE_SIZE; i++) {
        fin_not_probe >> usr_num >> mv_num >> date >> rate;
        not_probe(i, 0) = usr_num - 1;
        not_probe(i, 1) = mv_num - 1;
        not_probe(i, 2) = date;
        not_probe(i, 3) = rate;

        if ((i + 1) % 10000000 == 0) {
            std::cout << "." << std::flush;
        }
    }

    std::cout << ".\ndone.\n" << std::endl;
    fin_not_probe.close();
}

void fill_probe(Eigen::ArrayXXi& probe, std::string header) {
    std::cout << "Loading probe data..." << std::flush;
    std::ifstream fin_probe(header + "data/probe.dta");
    assert(fin_probe.is_open());
    assert(probe.rows() == PROBE_SIZE);
    int usr_num, mv_num, date, rate;

    for (int i = 0; i < PROBE_SIZE; i++) {
        fin_probe >> usr_num >> mv_num >> date >> rate;
        probe(i, 0) = usr_num - 1;
        probe(i, 1) = mv_num - 1;
        probe(i, 2) = date;
        probe(i, 3) = rate;
    }

    std::cout << "done.\n" << std::endl;
    fin_probe.close();
}

void fill_qual(Eigen::ArrayXXi& qual, std::string header) {
    std::cout << "Loading qual data..." << std::flush;
    std::ifstream fin_qual(header + "data/qual.dta");
    assert(fin_qual.is_open());
    assert(qual.rows() == QUAL_SIZE);
    int usr_num, mv_num, date;

    for (int i = 0; i < QUAL_SIZE; i++) {
        fin_qual >> usr_num >> mv_num >> date;
        qual(i, 0) = usr_num - 1;
        qual(i, 1) = mv_num - 1;
        qual(i, 2) = date;
    }

    std::cout << "done.\n" << std::endl;
    fin_qual.close();
}
