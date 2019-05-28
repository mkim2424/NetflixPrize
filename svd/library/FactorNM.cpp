#include "FactorNM.h"

FactorNM::FactorNM(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
    Eigen::ArrayXXi& qual, std::string header) : 
    Model(p, train, valid, qual, header) {
    preProcess();
}

void FactorNM::preProcess() {
    std::cout << "PREPROCESSING..." << std::flush;
    this->loc     = Eigen::VectorXi(M);
    this->X       = Eigen::ArrayXXd::Zero(N, k);
    this->Y       = Eigen::ArrayXXd::Zero(N, k);
    this->sum_X   = Eigen::ArrayXXd::Zero(M, k);
    this->sum_Y   = Eigen::ArrayXXd::Zero(M, k);
    this->R_u     = new std::vector<int>*[M];
    this->N_u     = new std::vector<int>*[M];
    this->ratings = new std::unordered_map<int, int>*[M];
    int curr_user = -1;

    for (int i = 0; i < train.rows(); i++) {
        Eigen::VectorXi row = train.row(i);
        int usr_num = row[0], mv_num = row[1], rate = row[3];

        if (usr_num == curr_user + 1) {
            loc[usr_num]     = i;
            R_u[usr_num]     = new std::vector<int>;
            N_u[usr_num]     = new std::vector<int>;
            ratings[usr_num] = new std::unordered_map<int, int>;
            curr_user = usr_num;
        }

        R_u[usr_num]->push_back(mv_num);
        (*ratings[usr_num])[mv_num] = rate;
    }

    // If train doesn't include probe, include movies watched in probe in N_u
    if (train.rows() == TRAINING_SIZE - PROBE_SIZE) {
        for (int i = 0; i < valid.rows(); i++) {
            Eigen::VectorXi row = valid.row(i);
            int usr_num = row[0], mv_num = row[1];
            N_u[usr_num]->push_back(mv_num);
        }
    } else {
        assert(train.rows() == TRAINING_SIZE);
    }

    for (int i = 0; i < qual.rows(); i++) {
        Eigen::VectorXi row = qual.row(i);
        int usr_num = row[0], mv_num = row[1];
        N_u[usr_num]->push_back(mv_num);
    }

    std::cout << "DONE.\n" << std::endl;
}

void FactorNM::update_sum_X(int usr_num) {
    Eigen::VectorXd x_sum = Eigen::VectorXd::Zero(k);

    for (int i = 0; i < (int) R_u[usr_num]->size(); i++) {
        int j = R_u[usr_num]->at(i);
        int r_uj = (*ratings[usr_num])[j];
        int b_uj = predictBias(usr_num, j);
        Eigen::VectorXd x_j = X.row(j);
        x_sum += (r_uj - b_uj) * x_j;
    }

    sum_X.row(usr_num) = x_sum;
}

void FactorNM::update_sum_Y(int usr_num) {
    Eigen::VectorXd y_sum = Eigen::VectorXd::Zero(k);

    for (int i = 0; i < (int) N_u[usr_num]->size(); i++) {
        int j = N_u[usr_num]->at(i);
        Eigen::VectorXd y_j = Y.row(j);
        y_sum += y_j;
    }

    sum_Y.row(usr_num) = y_sum;
}

void FactorNM::SGD(int epoch) {
    #pragma omp parallel for
    for (int curr_user = 0; curr_user < M; curr_user++) {
        int start_loc = loc[curr_user];          // location in training data
        int R_size    = R_u[curr_user]->size();
        int N_size    = N_u[curr_user]->size(); // number of mvs rated by curr_user
        double norm_R = 1.0 / sqrt(R_size);
        double norm_N = 1.0 / sqrt(N_size);

        update_sum_X(curr_user);
        update_sum_Y(curr_user);
        Eigen::VectorXd X_u = sum_X.row(curr_user);
        Eigen::VectorXd Y_u = sum_Y.row(curr_user);
        Eigen::VectorXd p_u = norm_R * X_u;

        if (R_size == 0) {
            std::cout << "ERROR: DIVIDE BY ZERO R" << std::endl;
        }

        if (N_size != 0) {
            p_u += norm_N * Y_u;
        }

        Eigen::VectorXd sum = Eigen::VectorXd::Zero(k);

        for (int j = 0; j < R_size; j++) {
            int curr_loc = start_loc + j;
            Eigen::VectorXi row = train.row(curr_loc);
            int usr_num = row[0], mv_num = row[1], date = row[2], rate = row[3];
            assert(usr_num == curr_user);
            double err = rate - predictValue(usr_num, mv_num, date);

            // Update baselines
            base_u[usr_num] += eta[0] * decay_factor *
                (err - reg[0] * base_u[usr_num]);
            base_i[mv_num]  += eta[1] * decay_factor *
                (err - reg[1] * base_i[mv_num]);

            // Update U and V
            Eigen::VectorXd old_p_u = U.row(usr_num);
            Eigen::VectorXd old_q_i = V.row(mv_num);
            Eigen::VectorXd new_p_u = 
                old_p_u + eta[2] * decay_factor * (err * old_q_i - reg[2] * old_p_u);
            Eigen::VectorXd new_q_i = old_q_i + eta[2] * decay_factor * 
                (err * p_u - reg[2] * old_q_i);
            U.row(usr_num) = new_p_u;
            V.row(mv_num) = new_q_i;

            // Keep track of err * norm * old_q_i for batch update
            sum += err * old_q_i;
        }

        for (int ind = 0; ind < R_size; ind++) {
            int j = R_u[curr_user]->at(ind);
            int r_uj = (*ratings[curr_user])[j];
            int b_uj = predictBias(curr_user, j);
            Eigen::VectorXd old_x_j = X.row(j);
            Eigen::VectorXd new_x_j = old_x_j + eta[2] * decay_factor * 
                (norm_R * (r_uj - b_uj) * sum - reg[2] * old_x_j);
            X.row(j) = new_x_j;
        }

        // Batch update y_j's for each user
        for (int ind = 0; ind < N_size; ind++) {
            int j = N_u[curr_user]->at(ind);
            Eigen::VectorXd old_y_j = Y.row(j);
            Eigen::VectorXd new_y_j = old_y_j + eta[2] * decay_factor *
                (norm_N * sum - reg[2] * old_y_j);
            Y.row(j) = new_y_j;
        }

        update_sum_X(curr_user);
        update_sum_Y(curr_user);
    }

    for (int usr_num = 0; usr_num < M; usr_num++) {
        update_sum_X(usr_num);
        update_sum_Y(usr_num);
    }
}

double FactorNM::predictBias(int usr_num, int mv_num) {
    return AVG_RATING + base_u[usr_num] + base_i[mv_num];
}

double FactorNM::predictValue(int usr_num, int mv_num, int date) {
    double b_u = base_u[usr_num];
    double b_i = base_i[mv_num];
    int R_size = R_u[usr_num]->size();
    int N_size = N_u[usr_num]->size();
    double norm_N = 0;

    if (R_size == 0) {
        std::cout << "ERROR: DIVIDE BY ZERO R" << std::endl;
    }

    if (N_size != 0) {
        norm_N = 1.0 / sqrt(N_size);
    }

    double norm_R = 1.0 / sqrt(R_size);
    Eigen::VectorXd p_u = U.row(usr_num);
    Eigen::VectorXd p_u2 = norm_R * sum_X.row(usr_num) + norm_N * sum_Y.row(usr_num);
    
    Eigen::VectorXd q_i = V.row(mv_num);

    return AVG_RATING + b_u + b_i + q_i.dot(p_u + p_u2);
}
