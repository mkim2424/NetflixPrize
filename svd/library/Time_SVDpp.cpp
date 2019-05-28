#include "Time_SVDpp.h"

Time_SVDpp::Time_SVDpp(Parameter p, Eigen::ArrayXXi& train, Eigen::ArrayXXi& valid,
    Eigen::ArrayXXi& qual, std::string header) : Model(p, train, valid, qual, header) {
    preProcess();
}

Time_SVDpp::~Time_SVDpp() {
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

void Time_SVDpp::preProcess() {
    std::cout << "PREPROCESSING..." << std::flush;
    base_i_bin  = Eigen::ArrayXXd::Zero(N, NUM_BINS);
    mean_date   = Eigen::VectorXd(M);
    alpha_u     = Eigen::VectorXd::Zero(M);
    alpha_p_u   = Eigen::ArrayXXd::Zero(M, k);
    dates_rated = new std::set<int> *[M];
    base_u_t    = new std::unordered_map<int, double> *[M];
    freq        = new std::unordered_map<int, int>    *[M];
    base_i_f    = Eigen::ArrayXXd::Zero(N, 10);
    
    // Getting mean date of rating
    Eigen::VectorXi total_date = Eigen::VectorXi::Zero(M);
    Eigen::VectorXi num_rating = Eigen::VectorXi::Zero(M);

    for (int usr_num = 0; usr_num < M; usr_num++) {
        dates_rated[usr_num] = new std::set<int>;
        base_u_t[usr_num]    = new std::unordered_map<int, double>;
        freq[usr_num]        = new std::unordered_map<int, int>;
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
        }

        (*freq[usr_num])[date]++;
    }

    for (int i = 0; i < M; i++) {
        mean_date[i] = (double) total_date[i] / (double) num_rating[i];
    }

    loc    = Eigen::VectorXi(M);
    Y      = Eigen::ArrayXXd::Zero(N, k);
    sum_Y  = Eigen::ArrayXXd::Zero(M, k);
    movies_watched = new std::vector<int>*[M];
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
            int usr_num = row[0], mv_num = row[1], date = row[2];
            movies_watched[usr_num]->push_back(mv_num);

            auto dates_rated_u = dates_rated[usr_num];

            if (dates_rated_u->find(date) == dates_rated_u->end()) {
                dates_rated_u->insert(date);
                (*base_u_t[usr_num])[date] = 0.0;
                (*freq[usr_num])[date] = 0;
            }

            (*freq[usr_num])[date]++;
        }
    } else {
        assert(train.rows() == TRAINING_SIZE);
    }

    for (int i = 0; i < qual.rows(); i++) {
        Eigen::VectorXi row = qual.row(i);
        int usr_num = row[0], mv_num = row[1], date = row[2];
        movies_watched[usr_num]->push_back(mv_num);

        auto dates_rated_u = dates_rated[usr_num];

        if (dates_rated_u->find(date) == dates_rated_u->end()) {
            dates_rated_u->insert(date);
            (*base_u_t[usr_num])[date] = 0.0;
            (*freq[usr_num])[date] = 0;
        }

        (*freq[usr_num])[date]++;
    }

    std::cout << "DONE.\n" << std::endl;
}

double Time_SVDpp::get_dev(int usr_num, int date) {
    double diff = date - mean_date[usr_num];
    return (diff > 0 ? 1.0 : -1.0) * pow(fabs(diff), 0.4);
}

int Time_SVDpp::lower_f(int usr_num, int date) {
    int F = (*freq[usr_num])[date];
    return (int) (log(F) / log(6.76));
}

void Time_SVDpp::update_sum_Y(int usr_num) {
    Eigen::VectorXd y_sum = Eigen::VectorXd::Zero(k);

    for (int i = 0; i < (int) movies_watched[usr_num]->size(); i++) {
        int j = movies_watched[usr_num]->at(i);
        Eigen::VectorXd y_j = Y.row(j);
        y_sum += y_j;
    }

    sum_Y.row(usr_num) = y_sum;
}

int Time_SVDpp::get_num_movies_with_rating(int usr_num) {
    int curr_loc = loc[usr_num];
    int next_loc = usr_num == M - 1 ? train.rows() : loc[usr_num + 1];
    return next_loc - curr_loc;
}

void Time_SVDpp::SGD(int epoch) {
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
            int bin = get_bin(date);
            double dev = get_dev(usr_num, date);

            // Update baselines
            base_u[usr_num] += eta[0] * decay_factor * 
                (err - reg[0] * base_u[usr_num]);
            base_i[mv_num]  += eta[1] * decay_factor * 
                (err - reg[1] * base_i[mv_num]);
            base_i_bin(mv_num, bin) += eta[3] * decay_factor * 
                (err - reg[3] * base_i_bin(mv_num, bin));
            (*base_u_t[usr_num])[date] += eta[5] * decay_factor *
                (err - reg[5] * (*base_u_t[usr_num])[date]);
            int low_f = lower_f(usr_num, date);
            base_i_f(mv_num, low_f) += eta[8] * decay_factor *
                (err - reg[8] * base_i_f(mv_num, low_f));

            alpha_u[usr_num] += eta[4] * decay_factor * 
                (err * dev - reg[4] * alpha_u[usr_num]);

            // Update U and V
            Eigen::VectorXd old_p_u = U.row(usr_num);
            Eigen::VectorXd old_q_i = V.row(mv_num);
            Eigen::VectorXd old_alpha_p_u = alpha_p_u.row(usr_num);

            Eigen::VectorXd new_p_u = old_p_u + eta[2] * decay_factor * 
                (err * old_q_i - reg[2] * old_p_u);

            Eigen::VectorXd new_q_i = old_q_i + eta[2] * decay_factor * 
                (err * (old_p_u + old_alpha_p_u * dev + norm * y_sum) - reg[2] * old_q_i);

            Eigen::VectorXd new_alpha_p_u = old_alpha_p_u + eta[9] * decay_factor * 
                (err * dev * old_q_i - reg[9] * old_alpha_p_u);

            U.row(usr_num) = new_p_u;
            V.row(mv_num)  = new_q_i;
            alpha_p_u.row(usr_num) = new_alpha_p_u;

            // Keep track of err * norm * old_q_i for batch update
            A += err * norm * old_q_i;
        }

        // Batch update y_j's for each user
        for (int ind = 0; ind < N_size; ind++) {
            int j = movies_watched[curr_user]->at(ind);
            Eigen::VectorXd old_y_j = Y.row(j);
            Eigen::VectorXd new_y_j = old_y_j + eta[2] * decay_factor * 
                (A - reg[2] * old_y_j);
            Y.row(j) = new_y_j;
        }

        update_sum_Y(curr_user);
    }

    for (int usr_num = 0; usr_num < M; usr_num++) {
        update_sum_Y(usr_num);
    }
}

double Time_SVDpp::predictValue(int usr_num, int mv_num, int date) {
    int bin    = get_bin(date);
    double dev = get_dev(usr_num, date);

    double b_u = base_u[usr_num] + alpha_u[usr_num] * dev + 
                 (*base_u_t[usr_num])[date];
    double b_i = base_i[mv_num] + base_i_bin(mv_num, bin) +
                 base_i_f(mv_num, lower_f(usr_num, date));

    Eigen::VectorXd p_u = U.row(usr_num) + alpha_p_u.row(usr_num) * dev;
    Eigen::VectorXd q_i = V.row(mv_num);
    Eigen::VectorXd y = sum_Y.row(usr_num);
    double norm = 1.0 / sqrt(movies_watched[usr_num]->size());
    double latent = q_i.dot(p_u + norm * y);

    return AVG_RATING + b_u + b_i + latent;
}
