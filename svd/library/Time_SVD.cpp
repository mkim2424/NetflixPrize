#include "Time_SVD.h"

Time_SVD::Time_SVD(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
    Eigen::ArrayXXi& qual, std::string header) : 
    Model(p, train, valid, qual, header) {}

Time_SVD::~Time_SVD() {
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

void Time_SVD::preProcess() {
    std::cout << "PREPROCESSING..." << std::endl;
    this->base_i_bin  = Eigen::ArrayXXd::Zero(N, NUM_BINS);
    this->mean_date   = Eigen::VectorXd(M);
    this->alpha_u     = Eigen::VectorXd::Zero(M);
    this->alpha_p_u   = Eigen::ArrayXXd::Zero(M, k);
    this->dates_rated = new std::set<int> *[M];
    this->base_u_t    = new std::unordered_map<int, double> *[M];
    this->freq        = new std::unordered_map<int, int>    *[M];
    this->base_i_f    = Eigen::ArrayXXd::Zero(N, 10);

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
        for (int i = 0; i < valid.rows(); i++) {
            Eigen::VectorXi row = valid.row(i);
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

double Time_SVD::get_dev(int usr_num, int date) {
    double diff = date - mean_date[usr_num];
    return (diff > 0 ? 1.0 : -1.0) * pow(fabs(diff), 0.4);
}

int Time_SVD::lower_f(int usr_num, int date) {
    int F = (*freq[usr_num])[date];
    return (int) (log(F) / log(6.76));
}

void Time_SVD::SGD(int epoch) {
    #pragma omp parallel for
    for (int i = 0; i < train.rows(); i++) {
        Eigen::VectorXi row = train.row(i);
        int usr_num = row[0], mv_num = row[1], date = row[2], rate = row[3];
        double err = rate - predictValue(usr_num, mv_num, date);
        int bin    = get_bin(date);
        double dev = get_dev(usr_num, date);
        int f      = lower_f(usr_num, date);

        if (f < 0) {
            std::cout << "ERRORRRRRR" << std::endl;
        }
        // double b_i = base_i[mv_num] + base_i_bin(mv_num, bin);
        // double c   = c_u[usr_num] + (*c_u_t[usr_num])[date];
        Eigen::VectorXd old_p_u = U.row(usr_num);
        Eigen::VectorXd old_q_i = V.row(mv_num);
        Eigen::VectorXd old_alpha_p_u = alpha_p_u.row(usr_num);

        base_u[usr_num] += eta[0] * decay_factor * (err - reg[0] * base_u[usr_num]);
        base_i[mv_num]  += eta[1] * decay_factor * (err - reg[1] * base_i[mv_num]);
        base_i_bin(mv_num, bin) +=
            eta[3] * decay_factor * (err - reg[3] * base_i_bin(mv_num, bin));
        (*base_u_t[usr_num])[date] += eta[5] * decay_factor *
            (err - reg[5] * (*base_u_t[usr_num])[date]);
        int low_f = lower_f(usr_num, date);
        base_i_f(mv_num, low_f) += eta[8] * decay_factor *
            (err - reg[8] * base_i_f(mv_num, low_f));

        alpha_u[usr_num] +=
            eta[4] * decay_factor * (err * dev - reg[4] * alpha_u[usr_num]);

        // c_u[usr_num] += ETA_C_U * decay_factor * 
        //     (err * b_i - REG_C_U * (c_u[usr_num] - 1));

        // (*c_u_t[usr_num])[date] += ETA_C_U_T * decay_factor * 
        //     (err * b_i - REG_C_U_T * (*c_u_t[usr_num])[date]);

        Eigen::VectorXd new_p_u = old_p_u + eta[2] * decay_factor * 
            (err * old_q_i - reg[2] * old_p_u);

        Eigen::VectorXd new_q_i = old_q_i + eta[2] * decay_factor * 
            (err * (old_p_u + old_alpha_p_u * dev) - reg[2] * old_q_i);

        Eigen::VectorXd new_alpha_p_u = old_alpha_p_u + eta[9] * decay_factor * 
            (err * dev * old_q_i - reg[9] * old_alpha_p_u);

        U.row(usr_num) = new_p_u;
        V.row(mv_num)  = new_q_i;
        alpha_p_u.row(usr_num) = new_alpha_p_u;
    }
}

double Time_SVD::predictValue(int usr_num, int mv_num, int date) {
    int bin    = get_bin(date);
    double dev = get_dev(usr_num, date);
    int f      = lower_f(usr_num, date);

    if (f < 0) {
        std::cout << "ERRORRRRRR" << std::endl;
    }

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
