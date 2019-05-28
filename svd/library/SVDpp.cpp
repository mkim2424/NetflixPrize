#include "SVDpp.h"

SVDpp::SVDpp(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
    Eigen::ArrayXXi& qual, std::string header) : 
    Model(p, train, valid, qual, header) {
    preProcess();
}

void SVDpp::preProcess() {
    this->loc    = Eigen::VectorXi(M);
    this->Y      = Eigen::ArrayXXd::Zero(N, k);
    this->sum_Y  = Eigen::ArrayXXd::Zero(M, k);
    this->movies_watched = new std::vector<int>*[M];
    int curr_user = -1;

    for (int i = 0; i < train.rows(); i++) {
        Eigen::VectorXi row = train.row(i);
        int usr_num = row[0], mv_num = row[1];

        if (usr_num == curr_user + 1) {
            loc[usr_num] = i;
            movies_watched[usr_num] = new std::vector<int>;
            curr_user = usr_num;
        }

        movies_watched[usr_num]->push_back(mv_num);
    }

    if (train.rows() == TRAINING_SIZE - PROBE_SIZE) {
        for (int i = 0; i < valid.rows(); i++) {
            Eigen::VectorXi row = valid.row(i);
            int usr_num = row[0], mv_num = row[1];
            movies_watched[usr_num]->push_back(mv_num);
        }
    } else {
        assert(train.rows() == TRAINING_SIZE);
    }

    for (int i = 0; i < qual.rows(); i++) {
        Eigen::VectorXi row = qual.row(i);
        int usr_num = row[0], mv_num = row[1];
        movies_watched[usr_num]->push_back(mv_num);
    }
}

void SVDpp::update_sum_Y(int usr_num) {
    Eigen::VectorXd y_sum = Eigen::VectorXd::Zero(k);

    for (int i = 0; i < (int) movies_watched[usr_num]->size(); i++) {
        int j = movies_watched[usr_num]->at(i);
        Eigen::VectorXd y_j = Y.row(j);
        y_sum += y_j;
    }

    sum_Y.row(usr_num) = y_sum;
}

int SVDpp::get_num_movies_with_rating(int usr_num) {
    int curr_loc = loc[usr_num];
    int next_loc = usr_num == M - 1 ? train.rows() : loc[usr_num + 1];
    return next_loc - curr_loc;
}

void SVDpp::SGD(int epoch) {
    #pragma omp parallel for
    for (int curr_user = 0; curr_user < M; curr_user++) {
        int start_loc = loc[curr_user];          // location in training data
        int N_size = movies_watched[curr_user]->size(); // number of mvs rated by curr_user
        double norm = 1.0 / sqrt(N_size);
        int num_movies_with_rating = get_num_movies_with_rating(curr_user);
        update_sum_Y(curr_user);
        Eigen::VectorXd y_sum = sum_Y.row(curr_user);
        Eigen::VectorXd A = Eigen::VectorXd::Zero(k);

        for (int j = 0; j < num_movies_with_rating; j++) {
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
            U.row(usr_num) += eta[2] * decay_factor * 
                (err * V.row(mv_num) - reg[2] * U.row(usr_num));
            Eigen::VectorXd new_q_i = old_q_i + eta[2] * decay_factor * 
                (err * (old_p_u + norm * y_sum) - reg[2] * old_q_i);
            V.row(mv_num) = new_q_i;

            // Keep track of err * norm * old_q_i for batch update
            A += err * norm * old_q_i;
        }

        // Batch update y_j's for each user
        for (int ind = 0; ind < N_size; ind++) {
            int j = movies_watched[curr_user]->at(ind);
            Eigen::VectorXd old_y_j = Y.row(j);
            Eigen::VectorXd new_y_j = old_y_j + eta[3] * decay_factor * 
                (A - reg[3] * old_y_j);
            Y.row(j) = new_y_j;
        }

        update_sum_Y(curr_user);
    }

    for (int usr_num = 0; usr_num < M; usr_num++) {
        update_sum_Y(usr_num);
    }
}

double SVDpp::predictValue(int usr_num, int mv_num, int date) {
    double b_u = base_u[usr_num];
    double b_i = base_i[mv_num];
    Eigen::VectorXd p_u = U.row(usr_num);
    Eigen::VectorXd q_i = V.row(mv_num);
    Eigen::VectorXd y = sum_Y.row(usr_num);
    double norm = 1.0 / sqrt(movies_watched[usr_num]->size());
    double latent = q_i.dot(p_u + norm * y);

    return AVG_RATING + b_u + b_i + latent;
}
