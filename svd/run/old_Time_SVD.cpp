#include "include.h"

#define ETA_BASE   0.01
#define ETA_LATENT 0.01
#define ETA_ALPHA  1e-5
#define ETA_ALPHA2 1e-5
// #define ETA_C_U    5.64e-3
// #define ETA_C_U_T  1.03e-3

#define REG_BASE   0.005
#define REG_LATENT 0.015
#define REG_ALPHA  50
#define REG_ALPHA2 70
#define REG_FREQ   1.1e-8
// #define REG_C_U    4.76e-2
// #define REG_C_U_T  1.90e-2

/*
 * DIFFERENCES:
 * starting with uniform distribution
 * calculating avg_rating on the spot, not using AVG_RATING
 */

std::string header = "/home/ec2-user/cs156b/svd/";
// std::string header = "/Users/jameswei/Documents/Caltech/cs156b/svd/";

int k;
int maxEpoch;

Eigen::ArrayXXi train;
Eigen::ArrayXXi probe;
Eigen::ArrayXXi qual;

Eigen::VectorXd base_u;
Eigen::VectorXd base_i;
Eigen::ArrayXXd U;
Eigen::ArrayXXd V;

Eigen::ArrayXXd base_i_bin;
Eigen::VectorXd mean_date;
Eigen::VectorXd alpha_u;
std::set<int>   **dates_rated;
std::unordered_map<int, double> **base_u_t;
std::unordered_map<int, int> **freq; // map btw date and number of ratings
Eigen::ArrayXXd base_i_f;
Eigen::VectorXd c_u;
std::unordered_map<int, double> **c_u_t;

Eigen::ArrayXXd alpha_p_u;

double get_dev(int usr_num, int date) {
    double diff = date - mean_date[usr_num];
    return (diff > 0 ? 1.0 : -1.0) * pow(fabs(diff), 0.4);
}

int lower_f(int usr_num, int date) {
    int F = (*freq[usr_num])[date];
    return (int) (log(F) / log(6.76));
}

void preProcess() {
    std::cout << "PREPROCESSING..." << std::flush;
    base_u      = Eigen::VectorXd::Zero(M);
    base_i      = Eigen::VectorXd::Zero(N);
    U           = Eigen::ArrayXXd::Random(M, k) / 100;
    V           = Eigen::ArrayXXd::Random(N, k) / 100;
    base_i_bin  = Eigen::ArrayXXd::Zero(N, NUM_BINS);
    mean_date   = Eigen::VectorXd(M);
    alpha_u     = Eigen::VectorXd::Zero(M);
    alpha_p_u   = Eigen::ArrayXXd::Zero(M, k);
    dates_rated = new std::set<int> *[M];
    base_u_t    = new std::unordered_map<int, double> *[M];
    freq        = new std::unordered_map<int, int>    *[M];
    base_i_f    = Eigen::ArrayXXd::Zero(N, 10);
    // c_u         = Eigen::VectorXd(M);
    // c_u_t       = new std::unordered_map<int, double> *[M];

    // for (int i = 0; i < M; i++) {
    //     c_u[i] = 1;
    // }

    // Getting mean date of rating
    Eigen::VectorXi total_date = Eigen::VectorXi::Zero(M);
    Eigen::VectorXi num_rating = Eigen::VectorXi::Zero(M);

    for (int usr_num = 0; usr_num < M; usr_num++) {
        dates_rated[usr_num] = new std::set<int>;
        base_u_t[usr_num]    = new std::unordered_map<int, double>;
        freq[usr_num]        = new std::unordered_map<int, int>;
        // c_u_t[usr_num]       = new std::unordered_map<int, double>;
    }

    for (int i = 0; i < train.rows(); i++) {
        Eigen::VectorXi row = train.row(i);
        int usr_num = row[0], date = row[2];

        total_date[usr_num] += date;
        num_rating[usr_num]++;

        auto dates_rated_u = dates_rated[usr_num];

        if (dates_rated_u->find(date) == dates_rated_u->end()) {
            dates_rated_u->insert(date);
            (*base_u_t[usr_num])[date] = 0.0;
            (*freq[usr_num])[date] = 0;
            // (*c_u_t[usr_num])[date] = 0.0;
        }

        (*freq[usr_num])[date]++;
    }

    for (int i = 0; i < M; i++) {
        mean_date[i] = (double) total_date[i] / (double) num_rating[i];
    }

    if (train.rows() == TRAINING_SIZE - PROBE_SIZE) {
        for (int i = 0; i < probe.rows(); i++) {
            Eigen::VectorXi row = probe.row(i);
            int usr_num = row[0], date = row[2];

            auto dates_rated_u = dates_rated[usr_num];

            if (dates_rated_u->find(date) == dates_rated_u->end()) {
                dates_rated_u->insert(date);
                (*base_u_t[usr_num])[date] = 0.0;
                (*freq[usr_num])[date] = 0;
                // (*c_u_t[usr_num])[date] = 0.0;
            }

            (*freq[usr_num])[date]++;
        }
    } else {
        assert(train.rows() == TRAINING_SIZE);
    }

    for (int i = 0; i < qual.rows(); i++) {
        Eigen::VectorXi row = qual.row(i);
        int usr_num = row[0], date = row[2];

        auto dates_rated_u = dates_rated[usr_num];

        if (dates_rated_u->find(date) == dates_rated_u->end()) {
            dates_rated_u->insert(date);
            (*base_u_t[usr_num])[date] = 0.0;
            (*freq[usr_num])[date] = 0;
            // (*c_u_t[usr_num])[date] = 0.0;
        }

        (*freq[usr_num])[date]++;
    }

    std::cout << "DONE.\n" << std::endl;
}

double predictValue(int usr_num, int mv_num, int date) {
    int bin    = get_bin(date);
    double dev = get_dev(usr_num, date);

    double b_u = base_u[usr_num] + alpha_u[usr_num] * dev + 
                 (*base_u_t[usr_num])[date];
    double b_i = base_i[mv_num] + base_i_bin(mv_num, bin) +
                 base_i_f(mv_num, lower_f(usr_num, date));
    // double c   = c_u[usr_num] + (*c_u_t[usr_num])[date];

    Eigen::VectorXd p_u = U.row(usr_num) + alpha_p_u.row(usr_num) * dev;
    Eigen::VectorXd q_i = V.row(mv_num);
    double latent       = p_u.dot(q_i);

    return AVG_RATING + b_u + b_i + latent;
}

double getValidError() {
    double error = 0;

    for (int i = 0; i < probe.rows(); i++) {
        Eigen::VectorXi row = probe.row(i);
        int usr_num = row[0], mv_num = row[1], date = row[2], rate = row[3];
        double err = rate - predictValue(usr_num, mv_num, date);
        error += pow(err, 2);
    }

    return sqrt(error / probe.rows());
}

void SGD(int epoch) {
    time_t before = time(0);
    std::cout << "EPOCH: " << epoch + 1 << "/" << maxEpoch << std::endl;
    int train_size = train.rows();
    double decay = pow(0.95, epoch);

    #pragma omp parallel for
    for (int i = 0; i < train_size; i++) {
        Eigen::VectorXi row = train.row(i);
        int usr_num = row[0], mv_num = row[1], date = row[2], rate = row[3];
        double err = rate - predictValue(usr_num, mv_num, date);
        int bin    = get_bin(date);
        double dev = get_dev(usr_num, date);
        // double b_i = base_i[mv_num] + base_i_bin(mv_num, bin);
        // double c   = c_u[usr_num] + (*c_u_t[usr_num])[date];
        Eigen::VectorXd old_p_u = U.row(usr_num);
        Eigen::VectorXd old_q_i = V.row(mv_num);
        Eigen::VectorXd old_alpha_p_u = alpha_p_u.row(usr_num);

        base_u[usr_num] += ETA_BASE * decay * (err - REG_BASE * base_u[usr_num]);
        base_i[mv_num]  += ETA_BASE * decay * (err - REG_BASE * base_i[mv_num]);
        base_i_bin(mv_num, bin) +=
            ETA_BASE * decay * (err - REG_BASE * base_i_bin(mv_num, bin));
        (*base_u_t[usr_num])[date] += ETA_BASE * decay *
            (err - REG_BASE * (*base_u_t[usr_num])[date]);
        int low_f = lower_f(usr_num, date);
        base_i_f(mv_num, low_f) += ETA_BASE * decay *
            (err - REG_FREQ * base_i_f(mv_num, low_f));

        alpha_u[usr_num] +=
            ETA_ALPHA * decay * (err * dev - REG_ALPHA * alpha_u[usr_num]);

        // c_u[usr_num] += ETA_C_U * decay * 
        //     (err * b_i - REG_C_U * (c_u[usr_num] - 1));

        // (*c_u_t[usr_num])[date] += ETA_C_U_T * decay * 
        //     (err * b_i - REG_C_U_T * (*c_u_t[usr_num])[date]);

        Eigen::VectorXd new_p_u = old_p_u + ETA_LATENT * decay * 
            (err * old_q_i - REG_LATENT * old_p_u);

        Eigen::VectorXd new_q_i = old_q_i + ETA_LATENT * decay * 
            (err * (old_p_u + old_alpha_p_u * dev) - REG_LATENT * old_q_i);

        Eigen::VectorXd new_alpha_p_u = old_alpha_p_u + ETA_ALPHA2 * decay * 
            (err * dev * old_q_i - REG_ALPHA2 * old_alpha_p_u);

        U.row(usr_num) = new_p_u;
        V.row(mv_num)  = new_q_i;
        alpha_p_u.row(usr_num) = new_alpha_p_u;
    }

    std::cout << "ERROR: " << getValidError() << std::endl;
    std::cout << "TOOK   " << time(0) - before << "s.\n" << std::endl;
}

void predictQual() {
    std::cout << "PREDICTING VALUES..." << std::flush;
    std::ofstream fout_qual(header + "results/Time_SVD_qual.dta");
    assert(fout_qual.is_open());

    for (int i = 0; i < QUAL_SIZE; i++) {
        Eigen::VectorXi row = qual.row(i);
        int usr_num = row[0], mv_num = row[1], date = row[2];
        fout_qual << clip(predictValue(usr_num, mv_num, date)) << "\n";
    }

    std::cout << "DONE." << std::endl;
}

void predictProbe() {
    std::cout << "PREDICTING VALUES..." << std::flush;
    std::ofstream fout_probe(header + "results/Time_SVD_probe.dta");
    assert(fout_probe.is_open());

    for (int i = 0; i < PROBE_SIZE; i++) {
        Eigen::VectorXi row = probe.row(i);
        int usr_num = row[0], mv_num = row[1], date = row[2];
        fout_probe << clip(predictValue(usr_num, mv_num, date)) << "\n";
    }
    
    std::cout << "DONE." << std::endl;
}

void clean() {
    for (int i = 0; i < M; i++) {
        delete dates_rated[i];
        delete base_u_t[i];
        delete freq[i];
        // delete c_u_t[i];
    }

    delete[] dates_rated;
    delete[] base_u_t;
    delete[] freq;
    // delete[] c_u_t;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "GIVE ME k AND maxEpochs." << std::endl;
        exit(EXIT_FAILURE);
    }

    k = std::atoi(argv[1]), maxEpoch = std::atoi(argv[2]);
    // train = Eigen::ArrayXXi(TRAINING_SIZE, 4);
    // fill_all(train, header);
    train = Eigen::ArrayXXi(TRAINING_SIZE - PROBE_SIZE, 4);
    fill_not_probe(train, header);
    probe = Eigen::ArrayXXi(PROBE_SIZE, 4);
    fill_probe(probe, header);
    qual  = Eigen::ArrayXXi(QUAL_SIZE, 3);
    fill_qual(qual, header);

    preProcess();

    std::cout << "TRAINING DATA WITH k = " << k << std::endl;

    for (int i = 0; i < maxEpoch; i++) {
        SGD(i);
    }

    std::cout << "DONE TRAINING DATA.\n" << std::endl;
    
    predictProbe();
    // predictQual();
    clean();
    std::cout << "\a";
    return 0;
}