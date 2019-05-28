
import numpy as np
# 102416306 - 2749898
NUM_TRAINING = 102416306 - 2749898
# 2749898
NUM_QUAL = 2749898

LAMBDA_1 = 25
LAMBDA_2 = 10

def get_bi (movie, mu, R_i_ratings):
    total = 0
    ratings = R_i_ratings[movie]
    for user_rating in ratings:
        total += user_rating - mu
    return total / (LAMBDA_1 + len(ratings))

def get_bu (user, mu, R_u_ratings, bi_dict, R_u):
    total = 0
    ratings = R_u_ratings[user]
    movies = R_u[user]
    for i in range(len(ratings)):
        total += ratings[i] - mu + bi_dict[movies[i]]

    return total / (LAMBDA_2 + len(ratings))



def main():
    print("Start Loading Data")
    #max_rows = 102416306
    # all_data = np.loadtxt('../../um_data/all.dta', max_rows = 100000, usecols = (0,1,3)).astype(int)

    training = np.empty([NUM_TRAINING, 3], dtype = int)
    qual = np.empty([NUM_QUAL, 3], dtype = int)
    row_training = 0
    row_qual = 0
    row_line = 0
    for line in open('../../um_data/all.dta'):
        if row_line % 10000000 == 0:
            print(str(row_line // 1000000), '%')
        row = np.fromstring(line, dtype=int, sep=' ')[[0,1,3]]
        if row[2] == 0:
            qual[row_qual] = row
            row_qual += 1
        else:
            training[row_training] = row
            row_training += 1
        row_line += 1

    # M = max(training[:,0]).astype(int) # users
    # print("Number of users: " + str(M))
    # N = max(training[:,1]).astype(int) # movies
    # print("Number of movies: " + str(N))


    print("Finished Loading Data")


    # training = []
    # qual = []
    # for row in all_data:
    #     if row[2] == 0:
    #         qual.append(row)
    #     else:
    #         training.append(row)
    #
    # training = np.array(training)
    # qual = np.array(qual)


    R_u_ratings = {} # set of items (ratings) rated by user u
    R_i_ratings = {} # set of users (ratings) who rated item i
    R_u = {}

    print("START filling up dictionaries")
    sum_ratings = 0

    row_count = 0
    for row in training:
        user = row[0]
        movie = row[1]
        rating = row[2]

        if row_count % 10000000 == 0:
            print(row_count)
        if user not in R_u:
            R_u_ratings[user] = [rating]
            R_u[user] = [movie]
        else:
            R_u_ratings[user].append(rating)
            R_u[user].append(movie)

        if movie not in R_i_ratings:
            R_i_ratings[movie] = [rating]
        else:
            R_i_ratings[movie].append(rating)
        sum_ratings += rating

        row_count += 1

    AVERAGE_RATING = sum_ratings / len(training)


    print("DONE filling up dictionaries")


    print("START filling up bi_dict")
    bi_dict = {}
    # make bi_dict
    for row in qual:
        movie = row[1]
        if movie not in bi_dict:
            # if movie not in R_i_ratings:
            #     bi_dict[movie] = 0
            bi_dict[movie] = get_bi (movie, AVERAGE_RATING, R_i_ratings)
    print("DONE filling up bi_dict")


    print("START making baseline predictions")
    #getting the ratings
    ratings = []
    for row in qual:
        user = row[0]
        movie = row[1]

        bi = bi_dict[movie]
        bu = get_bu(user, AVERAGE_RATING, R_u_ratings, bi_dict, R_u)

        ratings.append(AVERAGE_RATING + bi + bu)

    print("DONE making baseline predictions")

    ratings = np.clip(ratings, 1, 5)
    print("Length of qual: ", len(qual))
    print("Length of ratings: ", len(ratings))


    f = open("result_baseline.dta", "w+")
    for i in range(len(ratings)):
        f.write(str(ratings[i]) + '\n')
    f.close()



if __name__ == "__main__":
    main()
