
//Note: 
//	Most descriptive comments were added far after most code was implemented (for readability of TAs, etc.), so there may be some errors.
//	Several mlpack implementations were completey overwritten for other purposes (BiasSVDPolicy comes to mind) or substantially changed/debugged (RegSVDPolicy, SVDPlusPlusPolicy come to mind).
//	Code readablitly and organization has not been optimized almost whatsoever.
//	Many comments are reminders, for debugging, and for varying the functionality of the code. Hard coded parameters serve the latter purpose too. Descriptive comments usually describe the last apparent purpose of the corresponding code.

#include "stdafx.h"
//#include <mlpack/core.hpp>
//#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace mlpack;
using namespace mlpack::neighbor; // note NeighborSearch and NearestNeighborSort
using namespace mlpack::metric; 
using namespace mlpack::kmeans;
using namespace mlpack::cf;
using namespace mlpack::svd;
using namespace std;
using namespace arma;

//write a vector to file
void tof(string FILENAME, const vector<float> v)
{
	std::ofstream f(FILENAME);
	for (vector<float>::const_iterator i = v.begin(); i != v.end(); ++i) {
		f << *i << '\n';
	}
	f.close();
}

//write an arma vec to file
void tof(string FILENAME, const vec v)
{
	std::ofstream f(FILENAME);

	for (int i = 0; i < v.size(); i++) {
		f << v(i) << '\n';
	}

	f.close();
}

//read a db into an 2d array
int** fromf3(string FILENAME, int ROWS, int COLS)
{
	int** arr = new int*[ROWS];
	cout << "holy" << endl;
	ifstream myReadFile;
	myReadFile.open(FILENAME);

	for (int i = 0; i < ROWS; i++) {
		arr[i] = new int[COLS];
		for (int j = 0; j<COLS; j++) {
			myReadFile >> arr[i][j];
		}
		if (i % 1000000 == 0) {
			cout << i << endl;
		}
	}

	return arr;
}

//read a db into a 1d array
vector <int> fromfv(string FILENAME, int COLS)
{
	vector<int> data(COLS);

	ifstream myReadFile;
	myReadFile.open(FILENAME);
	cout << "COLS:\t" << COLS << endl;
	cout << "ready to read" << endl;
	getchar();
	int p;
	for (int i = 0; i < COLS; i++) {

		myReadFile >> p;// 
						//data[j][i]=p;
		if (i % 1000000 == 0) {
			cout << "ip" << endl;
			cout << i << endl;
			cout << p << endl;
		}
	}
	getchar();
	cout << "array read" << endl;
	getchar();
	return data;
}

//read a db into 2d vector array, preallocating efficiently
vector <vector <int>> fromf2(string FILENAME, int ROWS, int COLS)
{
	vector<vector<int>> data(COLS);
	for (int i = 0; i < COLS; i++)
	{
		data[i] = vector<int>(ROWS, 0);
	}

	ifstream myReadFile;
	myReadFile.open(FILENAME);
	cout << "ROWS:\t" << ROWS << endl;
	cout << "COLS:\t" << COLS << endl;
	cout << "Read start" << endl;
	//getchar();
	int p;
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j<COLS; j++) {
			myReadFile >> data[j][i];
		}
		if (i % 1000000 == 0) {
			//cout << "ip" << endl;
			cout << i << endl;
			//cout << p << endl;
		}
	}
	//getchar();
	cout << "Read end" << endl;
	//getchar();
	myReadFile.close();
	return data;
}

//read a db into 2d vector array of unknown rows, allocating dynamically
vector <vector <int>> fromf(string FILENAME, int COLS)
{
	fstream file;
	vector <vector <int>> arr; 
	vector <int> rowVector(COLS); 


	int row = 0; 

				 
	file.open(FILENAME, ios::in); 
	if (file.is_open()) { 
						  
		cout << "File correctly opened" << endl;
		
		while (file.good()) { 
			arr.push_back(rowVector); 
			for (int col = 0; col<COLS; col++) {
				file >> arr[row][col]; 
			}
			row++; 
			if (row % 1000000 == 0) {
				cout << row << endl;
			}
		}
	}
	else cout << "Unable to open file" << endl;
	file.close();
	return arr;
}

//perform a knn via overfit neighborhood svd with pearson distance, either for blending purposes or not
void doknn(bool forblend, string outfile)
{
	int alln = 102416306;
	alln = (int)(alln);
	string test1 = "C:/Users/DFCTech/Desktop/testo.dta";
	string test2 = "C:/Users/DFCTech/Desktop/testo.idx";

	string real1 = "C:/Users/DFCTech/Downloads/um/all.dta";
	string real2 = "C:/Users/DFCTech/Downloads/um/all.idx";


	vector <vector <int>> alld = fromf2(real1, alln, 4);//colum row
	cout << "file 1 read" << endl;
	vector <vector <int>> idxd = fromf2(real2, alln, 1);
	cout << "file 2 read" << endl;

	int probe_n = 0;
	int train_n = 0;
	int qual_n = 0;

	if (forblend) {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				probe_n++;
			}
			else if (idxd[0][r] == 5) {
				qual_n++;
			}
			else {
				train_n++;
			}
		}
	}
	else {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				probe_n++;
			}
			if (idxd[0][r] == 5) {
				qual_n++;
			}
			else {
				train_n++;
			}
		}
	}


	arma::u64_mat tellme(2, qual_n);

	arma::u64_mat tellme2(2, probe_n);

	arma::mat lookme(3, train_n);

	int n1 = 0;
	int n2 = 0;
	int n3 = 0;
	if (forblend) {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				tellme2(0, n3) = alld[0][r];
				tellme2(1, n3) = alld[1][r];
				n3++;
			}
			else if (idxd[0][r] == 5) {
				tellme(0, n1) = alld[0][r];
				tellme(1, n1) = alld[1][r];
				n1++;
			}
			else {
				lookme(0, n2) = alld[0][r];
				lookme(1, n2) = alld[1][r];
				lookme(2, n2) = alld[3][r];
				n2++;
			}
		}
	}
	else {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				tellme2(0, n3) = alld[0][r];
				tellme2(1, n3) = alld[1][r];
				n3++;
			}
			if (idxd[0][r] == 5) {
				tellme(0, n1) = alld[0][r];
				tellme(1, n1) = alld[1][r];
				n1++;
			}
			else {
				lookme(0, n2) = alld[0][r];
				lookme(1, n2) = alld[1][r];
				lookme(2, n2) = alld[3][r];
				n2++;
			}
		}
	}

	cout << "ready to learn" << endl;

	size_t neighborhood = 5;
	size_t rank = 20;

	clock_t t1, t2;

	t1 = clock();
	vec predictions1(qual_n);
	vec predictions2(probe_n);

	CFType<cf::RegSVDPolicy> cft(lookme, RegSVDPolicy(80, 0.015, 0.04), 1, 40, 80, 1e-5, false);

	//CFType<cf::BiasSVDPolicy> cf(lookme, BiasSVDPolicy(1, 0.02, 0.05, &lookme), 5, 1, 1, 1000, false);
	cout << "svd s" << endl;

	


	//CFType<cf::SVDPlusPlusPolicy> cf(lookme);
	t2 = clock();

	cout << "train time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;


	std::cout << "model trained" << std::endl;

	//cout << "!";
	//cout << cf.Predict<cf::PearsonSearch>(tellme(0, 0), tellme(1, 0)) << endl;

	
	t1 = clock();

	if (forblend) {
		cft.Predict<PearsonSearch>(tellme2, predictions2);
	}
	else {

		cft.Predict<PearsonSearch>(tellme, predictions1);
	}
	t2 = clock();
	cout << "predict time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;

	std::cout << "model evaluated" << std::endl;
	if (forblend) {
		tof(outfile, predictions2);
	}
	else {
		tof(outfile, predictions1);
	}

	std::cout << "results exported" << std::endl;
}

//perform a regularized svd with various parameters (specified in line), either for blending purposes or not, using some fraction of the reference set
void dosvd(bool forblend, string outfile)
{
	int alln = 102416306;
	alln = (int)(alln);
	string test1 = "C:/Users/DFCTech/Desktop/testo.dta";
	string test2 = "C:/Users/DFCTech/Desktop/testo.idx";

	string real1 = "C:/Users/DFCTech/Downloads/um/all.dta";
	string real2 = "C:/Users/DFCTech/Downloads/um/all.idx";


	vector <vector <int>> alld = fromf2(real1, alln, 4);//colum row
	cout << "file 1 read" << endl;
	vector <vector <int>> idxd = fromf2(real2, alln, 1);
	cout << "file 2 read" << endl;

	int probe_n = 0;
	int train_n = 0;
	int qual_n = 0;

	if (forblend) {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				probe_n++;
			}
			else if (idxd[0][r] == 5) {
				qual_n++;
			}
			else {
				train_n++;
			}
		}
	}
	else {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				probe_n++;
			}
			if (idxd[0][r] == 5) {
				qual_n++;
			}
			else {
				train_n++;
			}
		}
	}


	arma::u64_mat tellme(2, qual_n);

	arma::u64_mat tellme2(2, probe_n);

	arma::mat lookme(3, train_n);

	int n1 = 0;
	int n2 = 0;
	int n3 = 0;
	if (forblend) {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				tellme2(0, n3) = alld[0][r];
				tellme2(1, n3) = alld[1][r];
				n3++;
			}
			else if (idxd[0][r] == 5) {
				tellme(0, n1) = alld[0][r];
				tellme(1, n1) = alld[1][r];
				n1++;
			}
			else {
				lookme(0, n2) = alld[0][r];
				lookme(1, n2) = alld[1][r];
				lookme(2, n2) = alld[3][r];
				n2++;
			}
		}
	}
	else {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				tellme2(0, n3) = alld[0][r];
				tellme2(1, n3) = alld[1][r];
				n3++;
			}
			if (idxd[0][r] == 5) {
				tellme(0, n1) = alld[0][r];
				tellme(1, n1) = alld[1][r];
				n1++;
			}
			else {
				lookme(0, n2) = alld[0][r];
				lookme(1, n2) = alld[1][r];
				lookme(2, n2) = alld[3][r];
				n2++;
			}
		}
	}

	cout << "ready to learn" << endl;

	size_t neighborhood = 5;
	size_t rank = 20;

	clock_t t1, t2;

	t1 = clock();
	vec predictions1(qual_n);
	vec predictions2(probe_n);

	CFType<cf::RegSVDPolicy> cft(lookme, RegSVDPolicy(80), 5, 30, 80, 1e-5, false);

	//CFType<cf::BiasSVDPolicy> cf(lookme, BiasSVDPolicy(1, 0.02, 0.05, &lookme), 5, 1, 1, 1000, false);
	cout << "svd s" << endl;




	//CFType<cf::SVDPlusPlusPolicy> cf(lookme);
	t2 = clock();

	cout << "train time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;


	std::cout << "model trained" << std::endl;

	//cout << "!";
	//cout << cf.Predict<cf::PearsonSearch>(tellme(0, 0), tellme(1, 0)) << endl;


	t1 = clock();

	if (forblend) {
		cft.Predict<PearsonSearch>(tellme2, predictions2);
	}
	else {

		cft.Predict<PearsonSearch>(tellme, predictions1);
	}
	t2 = clock();
	cout << "predict time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;

	std::cout << "model evaluated" << std::endl;
	if (forblend) {
		tof(outfile, predictions2);
	}
	else {
		tof(outfile, predictions1);
	}

	std::cout << "results exported" << std::endl;
}

//perform one of various knn methods (those not used are commented out) either for blending purposes or not, using some fraction of the reference set
void doknn2(bool forblend, string outfile)
{
	int alln = 102416306;
	alln = (int)(alln);
	string test1 = "C:/Users/DFCTech/Desktop/testo.dta";
	string test2 = "C:/Users/DFCTech/Desktop/testo.idx";

	string real1 = "C:/Users/DFCTech/Downloads/um/all.dta";
	string real2 = "C:/Users/DFCTech/Downloads/um/all.idx";


	vector <vector <int>> alld = fromf2(real1, alln, 4);//colum row
	cout << "file 1 read" << endl;
	vector <vector <int>> idxd = fromf2(real2, alln, 1);
	cout << "file 2 read" << endl;

	int probe_n = 0;
	int train_n = 0;
	int qual_n = 0;

	if (forblend) {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				probe_n++;
			}
			else if (idxd[0][r] == 5) {
				qual_n++;
			}
			else {
				train_n++;
			}
		}
	}
	else {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				probe_n++;
			}
			if (idxd[0][r] == 5) {
				qual_n++;
			}
			else {
				train_n++;
			}
		}
	}

	int d = 0;
	if (forblend) {
		d = probe_n;
	}
	else {
		d = qual_n;
	}

	arma::mat tellme(3, d);

	arma::mat lookme(3, train_n);
	arma:vec ratings(train_n);

	int n1 = 0;
	int n2 = 0;
	int n3 = 0;
	if (forblend) {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				tellme(0, n3) = alld[0][r];
				tellme(1, n3) = alld[1][r];
				tellme(1, n3) = alld[2][r];
				n3++;
			}
			else if (idxd[0][r] == 5) {
			}
			else {
				lookme(0, n2) = alld[0][r];
				lookme(1, n2) = alld[1][r];
				lookme(2, n2) = alld[2][r];
				ratings(n2) = alld[3][r];
				n2++;
			}
		}
	}
	else {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
			}
			if (idxd[0][r] == 5) {
				tellme(0, n1) = alld[0][r];
				tellme(1, n1) = alld[1][r];
				tellme(2, n1) = alld[1][r];
				n1++;
			}
			else {
				lookme(0, n2) = alld[0][r];
				lookme(1, n2) = alld[1][r];
				lookme(2, n2) = alld[2][r];
				ratings(n2) = alld[3][r];
				n2++;
			}
		}
	}
	int maxn = 500000;

	mat lookme_s(3, maxn);
	vec ratings_s(maxn);
	for (int i = 0; i < maxn; i++) {
		int r = rand() % lookme.n_cols;
		lookme_s(0, i) = lookme(0, r);
		lookme_s(1, i) = lookme(1, r);
		lookme_s(2, i) = lookme(2, r);
		ratings_s(i) = ratings(r);
	}
	
	arma::mat tellme_N;
	tellme_N = arma::normalise(tellme.each_row() - arma::mean(tellme));
	

	arma::mat lookme_N(arma::size(lookme_s));
	lookme_N = arma::normalise(
		lookme_s.each_row() - arma::mean(lookme_s));

	Mat<size_t> neighborhood;
	mat similarities;
	clock_t t1, t2;
	std::cout << "m1" << endl;
	t1 = clock();

	//RASearch<> a(lookme_N);
	//KNN a;
	//a.Epsilon() = 0.25;
	//a.Train(lookme_N);
	RASearch<> a(lookme_N, true, true, 5, 0.75, false, false, 20);
	std::cout << "m1" << endl;
	a.Search(tellme_N, 5, neighborhood, similarities);
	t2 = clock();
	std::cout << "predict time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;
	//cout << similarities << endl;
	//std::cout << neighborhood << endl;
	//std::cin.get();

	vec predictions(tellme_N.n_cols);

	for (int u = 0; u < neighborhood.n_cols; u++) {
		double rating = 0;
		for (int n = 0; n < neighborhood.n_rows; n++) {
			int db = neighborhood(n, u);
			if (db >= ratings_s.n_elem || db < 0) {
				rating += 3;
				cout << db << "\ter"<< endl;
			}
			else
				rating += ratings_s(neighborhood(n, u));
		}
		predictions(u) = rating / neighborhood.n_rows;
	}

	std::cout << "ready to learn" << endl;

	tof(outfile, predictions);

	std::cout << "results exported" << std::endl;
}

//perform a knn svd hybrid method either for blending purposes or not, using some fraction of the reference set
void doknn2svd(bool forblend, string outfile)
{
	int alln = 102416306;
	alln = (int)(alln);
	string test1 = "C:/Users/DFCTech/Desktop/testo.dta";
	string test2 = "C:/Users/DFCTech/Desktop/testo.idx";

	string real1 = "C:/Users/DFCTech/Downloads/um/all.dta";
	string real2 = "C:/Users/DFCTech/Downloads/um/all.idx";


	vector <vector <int>> alld = fromf2(real1, alln, 4);//colum row
	cout << "file 1 read" << endl;
	vector <vector <int>> idxd = fromf2(real2, alln, 1);
	cout << "file 2 read" << endl;

	int probe_n = 0;
	int train_n = 0;
	int qual_n = 0;

	if (forblend) {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				probe_n++;
			}
			else if (idxd[0][r] == 5) {
				qual_n++;
			}
			else {
				train_n++;
			}
		}
	}
	else {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				probe_n++;
			}
			if (idxd[0][r] == 5) {
				qual_n++;
			}
			else {
				train_n++;
			}
		}
	}

	int d = 0;
	if (forblend) {
		d = probe_n;
	}
	else {
		d = qual_n;
	}

	arma::mat tellme(3, d);

	arma::mat lookme(3, train_n);
arma:vec ratings(train_n);

	int n1 = 0;
	int n2 = 0;
	int n3 = 0;
	if (forblend) {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				tellme(0, n3) = alld[0][r];
				tellme(1, n3) = alld[1][r];
				tellme(1, n3) = alld[2][r];
				n3++;
			}
			else if (idxd[0][r] == 5) {
			}
			else {
				lookme(0, n2) = alld[0][r];
				lookme(1, n2) = alld[1][r];
				lookme(2, n2) = alld[2][r];
				ratings(n2) = alld[3][r];
				n2++;
			}
		}
	}
	else {
		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
			}
			if (idxd[0][r] == 5) {
				tellme(0, n1) = alld[0][r];
				tellme(1, n1) = alld[1][r];
				tellme(2, n1) = alld[1][r];
				n1++;
			}
			else {
				lookme(0, n2) = alld[0][r];
				lookme(1, n2) = alld[1][r];
				lookme(2, n2) = alld[2][r];
				ratings(n2) = alld[3][r];
				n2++;
			}
		}
	}
	CFType<cf::RegSVDPolicy> cf(lookme, RegSVDPolicy(80, 0.015, 0.04), 1, 80, 200, 1e-5, false);
	mat lookme2(3, lookme.n_cols+tellme.n_cols);
	vec ratings2(lookme.n_cols + tellme.n_cols);
	for (int i = 0; i < lookme2.n_cols; i++) {
		if (i < lookme.n_cols) {
			lookme2(0, i) = lookme(0, i);
			lookme2(1, i) = lookme(1, i);
			lookme2(2, i) = lookme(2, i);
			ratings2(i) = ratings(i);
		}
		else {
			int j = i - lookme.n_cols;
			lookme2(0, i) = tellme(0, j);
			lookme2(1, i) = lookme(1, j);
			lookme2(2, i) = lookme(2, j);
			ratings2(i) = cf.Predict(lookme2(0,i), lookme2(1, i));
		}
		

	}

	int maxn = 500000;

	mat lookme_s(3, maxn);
	vec ratings_s(maxn);
	for (int i = 0; i < maxn; i++) {
		int r = rand() % lookme.n_cols;
		lookme_s(0, i) = lookme(0, r);
		lookme_s(1, i) = lookme(1, r);
		lookme_s(2, i) = lookme(2, r);
		ratings_s(i) = ratings(r);
	}

	arma::mat tellme_N;
	tellme_N = arma::normalise(tellme.each_row() - arma::mean(tellme));


	arma::mat lookme_N(arma::size(lookme_s));
	lookme_N = arma::normalise(
		lookme_s.each_row() - arma::mean(lookme_s));

	Mat<size_t> neighborhood;
	mat similarities;
	clock_t t1, t2;
	std::cout << "m1" << endl;
	t1 = clock();

	//RASearch<> a(lookme_N);
	//KNN a;
	//a.Epsilon() = 0.25;
	//a.Train(lookme_N);
	RASearch<> a(lookme_N, true, true, 5, 0.75, false, false, 20);
	std::cout << "m1" << endl;
	a.Search(tellme_N, 5, neighborhood, similarities);
	t2 = clock();
	std::cout << "predict time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;
	//cout << similarities << endl;
	//std::cout << neighborhood << endl;
	//std::cin.get();

	vec predictions(tellme_N.n_cols);

	for (int u = 0; u < neighborhood.n_cols; u++) {
		double rating = 0;
		for (int n = 0; n < neighborhood.n_rows; n++) {
			int db = neighborhood(n, u);
			if (db >= ratings_s.n_elem || db < 0) {
				rating += 3;
				cout << db << "\ter" << endl;
			}
			else
				rating += ratings_s(neighborhood(n, u));
		}
		predictions(u) = rating / neighborhood.n_rows;
	}

	std::cout << "ready to learn" << endl;

	tof(outfile, predictions);

	std::cout << "results exported" << std::endl;
}

int main()
{
	//for i/o optimization
	std::ios_base::sync_with_stdio(false);

	//various tests, models, and procedures are chosen based their number in this switch statement. A comment before each vaguely attempts to describe what each case does.
	switch (10) {
	
	//very qick svd and pearson search test
	case 1: {
		int alln = 102416306;
		alln = (int)(alln);
		string test1 = "C:/Users/DFCTech/Desktop/testo.dta";
		string test2 = "C:/Users/DFCTech/Desktop/testo.idx";

		string real1 = "C:/Users/DFCTech/Downloads/um/all.dta";
		string real2 = "C:/Users/DFCTech/Downloads/um/all.idx";
		//vector <int> alld = fromfv(real1, alln);
		//vector <vector<int>> alld = fromf2(real1, alln,4);

		vector <vector <int>> alld = fromf2(real1, alln, 4);//colum row
		cout << "file 1 read" << endl;
		vector <vector <int>> idxd = fromf2(real2, alln, 1);
		cout << "file 2 read" << endl;
		int bn1 = 2749898;
		int bn2 = 102416306 - bn1;
		arma::u64_mat tellme(2, bn1);

		bn2 = (int)(bn2);
		arma::mat lookme(3, alln);
		cout << alld[0].size() << endl;
		cout << "what was it" << endl;
		//cin.get();
		int n1 = 0;
		int n2 = 0;

		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 5) {
				tellme(0, n1) = alld[0][r];
				tellme(1, n1) = alld[1][r];
				n1++;

				lookme(0, n2) = alld[0][r];
				lookme(1, n2) = alld[1][r];
				lookme(2, n2) = 3;
				n2++;
			}
			else {
				lookme(0, n2) = alld[0][r];
				lookme(1, n2) = alld[1][r];
				lookme(2, n2) = alld[3][r];

				n2++;
				//if (n2 >= bn2)break;
			}
		}
		cout << n1 << endl;
		cout << n2 << endl;
		cout << "ready to learn" << endl;
		//cin.get();

		size_t neighborhood = 5;
		// The rank of the decomposition.
		size_t rank = 20;

		//arma::Mat<size_t> recommendations; // Recommendations
		//svd::
		//svd::
		//CFType<> cf(lookme);
		//cf::
		clock_t t1, t2;

		cout << "size\t" << bn2 << endl;
		t1 = clock();
		//CFType<cf::SVDPlusPlusPolicy> cf(lookme,SVDPlusPlusPolicy(20,0.001,0.1),5,20,20,1e-5,false);
		CFType<cf::RegSVDPolicy> cf(lookme, RegSVDPolicy(20), 2, 20, 20, 1e-5, false);
		cout << "bias" << endl;
		//CFType<cf::BiasSVDPolicy> cf(lookme, BiasSVDPolicy(20,0.02,0.05), 5, 20, 20, 1e-5, false);


		//CFType<cf::SVDPlusPlusPolicy> cf(lookme);
		t2 = clock();

		cout << "train time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;
		//CFType<svd::RegularizedSVD<>> cf(lookme, svd::RegularizedSVD<>(100, 0.01, 0.1), neighborhood, rank);
		//CFType<> cf(lookme, svd::RegularizedSVD<>(200, 0.01, 0.1), neighborhood, rank);
		//CFType cf(lookme, svd::RegularizedSVD<>(200, 0.01, 0.1), neighborhood, rank);
		//CFType cf(lookme, amf::NMFALSFactorizer(), neighborhood, rank);

		std::cout << "model trained" << std::endl;
		/*
		vector<float> v;
		for (int i = 0; i < lookme.n_cols; i++) {
		v.push_back((float)cf.Predict(tellme(0, i), tellme(1, i)));
		}
		*/
		cout << "!";
		cout << cf.Predict<cf::PearsonSearch>(tellme(0, 0), tellme(1, 0)) << endl;
		vec predictions(bn1);
		t1 = clock();
		cf.Predict(tellme, predictions);
		t2 = clock();
		cout << "predict time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;

		std::cout << "model evaluated" << std::endl;
		string outfile = "C:/Users/DFCTech/Desktop/out.dta";
		tof(outfile, predictions);
		std::cout << "results exported" << std::endl;
		cin.get();

		break;
	}
	//bias svd
	case 2: {
		int alln = 102416306;
		alln = (int)(alln / 100);
		string test1 = "C:/Users/DFCTech/Desktop/testo.dta";
		string test2 = "C:/Users/DFCTech/Desktop/testo.idx";

		string real1 = "C:/Users/DFCTech/Downloads/um/all.dta";
		string real2 = "C:/Users/DFCTech/Downloads/um/all.idx";
		//vector <int> alld = fromfv(real1, alln);
		//vector <vector<int>> alld = fromf2(real1, alln,4);

		vector <vector <int>> alld = fromf2(real1, alln, 4);//colum row
		cout << "file 1 read" << endl;
		vector <vector <int>> idxd = fromf2(real2, alln, 1);
		cout << "file 2 read" << endl;
		int bn1 = 2749898;
		int bn2 = 102416306 - bn1;
		arma::u64_mat tellme(2, bn1);

		bn2 = (int)(bn2 / 1000);
		arma::mat lookme(3, bn2);
		cout << alld[0].size() << endl;
		cout << "what was it" << endl;
		//cin.get();
		int n1 = 0;
		int n2 = 0;

		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 5) {
				tellme(0, n1) = alld[0][r];
				tellme(1, n1) = alld[1][r];
				n1++;
			}
			else {
				if (n2 < bn2) {
					lookme(0, n2) = alld[0][r];
					lookme(1, n2) = alld[1][r];
					lookme(2, n2) = alld[3][r];

					n2++;
				}

			}
		}
		cout << n1 << endl;
		cout << n2 << endl;
		cout << "ready to learn" << endl;
		//cin.get();

		size_t neighborhood = 5;
		// The rank of the decomposition.
		size_t rank = 20;

		//arma::Mat<size_t> recommendations; // Recommendations
		//svd::
		//svd::
		//CFType<> cf(lookme);
		//cf::
		clock_t t1, t2;

		cout << "size\t" << bn2 << endl;
		t1 = clock();
		//CFType<cf::SVDPlusPlusPolicy> cf(lookme,SVDPlusPlusPolicy(20,0.001,0.1),5,20,20,1e-5,true);
		//CFType<cf::RegSVDPolicy> cf(lookme, RegSVDPolicy(80), 5, 20, 80, 1e-5, false);
		//CFType<cf::BatchSVDPolicy> cf(lookme);

		CFType<cf::BiasSVDPolicy> cf(lookme, BiasSVDPolicy(10, 0.02, 0.005), 5, 20, 10, 1e-5, false);


		//CFType<cf::SVDPlusPlusPolicy> cf(lookme);
		t2 = clock();

		cout << "train time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;
		//CFType<svd::RegularizedSVD<>> cf(lookme, svd::RegularizedSVD<>(100, 0.01, 0.1), neighborhood, rank);
		//CFType<> cf(lookme, svd::RegularizedSVD<>(200, 0.01, 0.1), neighborhood, rank);
		//CFType cf(lookme, svd::RegularizedSVD<>(200, 0.01, 0.1), neighborhood, rank);
		//CFType cf(lookme, amf::NMFALSFactorizer(), neighborhood, rank);

		std::cout << "model trained" << std::endl;
		/*
		vector<float> v;
		for (int i = 0; i < lookme.n_cols; i++) {
		v.push_back((float)cf.Predict(tellme(0, i), tellme(1, i)));
		}
		*/

		vec predictions(bn1);
		t1 = clock();
		cf.Predict(tellme, predictions);
		t2 = clock();
		cout << "predict time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;

		std::cout << "model evaluated" << std::endl;
		string outfile = "C:/Users/DFCTech/Desktop/out.dta";
		tof(outfile, predictions);
		std::cout << "results exported" << std::endl;
		cin.get();

		break;
	}
	//time binned svd
	case 4: {
		int parts = 10;
		int mintime = 1;
		int maxtime = 2243;

		int alln = 102416306;//total number of observations, known or not
		int bn1 = 2749898;//number not
		int bn2 = 102416306 - bn1;//number known
		int proben = 1374949;//number in known represenative part

		string real1 = "C:/Users/DFCTech/Downloads/um/all.dta";
		string real2 = "C:/Users/DFCTech/Downloads/um/all.idx";

		cout << "begin: file read" << endl;
		vector <vector <int>> alld = fromf2(real1, alln, 4);//colum row
		vector <vector <int>> idxd = fromf2(real2, alln, 1);
		cout << "end: file read" << endl;

		vec overallpredictions(bn1);
		vec overallpredictions2(proben);

		for (int p = 0; p < parts; p++) {
			cout << "begin: part " << p << endl;

			//count the number of known,unknown,probe that fall in time
			int nknown = 0;
			int nunknown = 0;
			int nprobe = 0;
			for (int i = 0; i < alln; i++) {
				int a = floor(parts*((double)(alld[2][i] - mintime)) / (maxtime - mintime));
				if (a == p) {
					if (idxd[0][i] == 4) {
						nprobe++;
					}
					if (idxd[0][i] == 5) {
						nunknown++;
					}
					else {
						nknown++;
					}
				}
			}

			//put these known,unknown,probe  in a matrix while remembering the order within e.g. unknown/probe
			arma::mat lookme(3, nknown);

			arma::u64_mat tellme(2, nunknown);
			arma::u64_mat tellme2(2, nprobe);

			vec predictions(nunknown);
			vec predictions2(nprobe);

			vector<int> is(nunknown);
			vector<int> is2(nprobe);

			int iknown = 0;
			int iunknown = 0;
			int iprobe = 0;

			int iunknown2 = 0;
			int iprobe2 = 0;
			for (int i = 0; i < alln; i++) {
				int a = floor(parts*((double)(alld[2][i] - mintime)) / (maxtime - mintime));

				if (idxd[0][i] == 4) {
					if (a == p) {
						tellme2(0, iprobe) = alld[0][i];
						tellme2(1, iprobe) = alld[1][i];
						is2[iprobe] = iprobe2;
						iprobe++;
					}
					iprobe2++;
				}
				if (idxd[0][i] == 5) {
					if (a == p) {
						tellme(0, iunknown) = alld[0][i];
						tellme(1, iunknown) = alld[1][i];
						is[iunknown] = iunknown2;
						iunknown++;
					}
					iunknown2++;
				}
				else {
					if (a == p) {
						lookme(0, iknown) = alld[0][i];
						lookme(1, iknown) = alld[1][i];
						lookme(2, iknown) = alld[3][i];
						iknown++;
					}
				}

			}

			//train on the known
			cout << "begin: train " << p << endl;
			CFType<cf::RegSVDPolicy> cf(lookme, RegSVDPolicy(40), 5, 20, 40, 1e-5, false);//rank,iterations
			cout << "end: train " << p << endl;

			//evaluate on the unknown,probe
			cout << "begin: evaluate " << p << endl;
			cf.Predict(tellme, predictions);
			cf.Predict(tellme2, predictions2);
			cout << "end: evaluate " << p << endl;

			//use order memory to put evalaution in right place
			for (int i = 0; i < nunknown; i++) {
				overallpredictions(is[i]) = predictions(i);
			}
			for (int i = 0; i < nprobe; i++) {
				overallpredictions2(is2[i]) = predictions2(i);
			}
			cout << "end: part " << p << endl;
		}
		//write to file
		string outfile = "C:/Users/DFCTech/Desktop/out_quiz.dta";
		tof(outfile, overallpredictions);

		string outfile2 = "C:/Users/DFCTech/Desktop/out_probe.dta";
		tof(outfile2, overallpredictions2);

		cin.get();
		break;
	}
	//NMF testing on simulated data and random subsets (subsets controlled by commenting)
	case 6:
	{
		/*	
		int r_alln = 102416306;
			string test1 = "C:/Users/DFCTech/Desktop/testo.dta";
			string test2 = "C:/Users/DFCTech/Desktop/testo.idx";

			string real1 = "C:/Users/DFCTech/Downloads/um/all.dta";
			string real2 = "C:/Users/DFCTech/Downloads/um/all.idx";
			
			int alln = (int)(r_alln);

			vector <vector <int>> alld = fromf2(real1, r_alln, 4);//colum row
			cout << "file 1 read" << endl;
			vector <vector <int>> idxd = fromf2(real2, r_alln, 1);
			cout << "file 2 read" << endl;*/
			/*
			vector<int> ri(r_alln);
			for (int i = 0; i < r_alln; i++)
			{
				ri[i] = i;
			}
			
			
			random_shuffle(ri.begin(), ri.end());

			vector<vector<int>> alld(4);
			for (int i = 0; i < 4; i++)
			{
				alld[i] = vector<int>(alln, 0);
			}
			vector<vector<int>> idxd(1);
			for (int i = 0; i < 1; i++)
			{
				idxd[i] = vector<int>(alln, 0);
			}
			
			for (int i = 0; i < alln; i++) {
				for (int j = 0; j < 4; j++)
				{
					alld[j][i] = r_alld[j][ri[i]];
				}
				for (int j = 0; j < 1; j++)
				{
					idxd[j][i] = r_idxd[j][ri[i]];
				}
				
			}
			*/
		/*
			int probe_n = 0;
			int train_n = 0;
			int qual_n = 0;

			for (int r = 0; r < alln; r++) {
				if (idxd[0][r] == 4) {
					probe_n++;
				}
				if (idxd[0][r] == 5) {
					qual_n++;
				}
				else {
					train_n++;
				}
			}*/
			
/*		
			arma::u64_mat tellme(2, qual_n);

			arma::u64_mat tellme2(2, probe_n);

			arma::mat lookme(3, train_n);

			int n1 = 0;
			int n2 = 0;
			int n3 = 0;

			for (int r = 0; r < alln; r++) {
				if (idxd[0][r] == 4) {
					tellme2(0, n3) = alld[0][r];
					tellme2(1, n3) = alld[1][r];
					n3++;
				}
				if (idxd[0][r] == 5) {
					tellme(0, n1) = alld[0][r];
					tellme(1, n1) = alld[1][r];
					n1++;
				}
				else {
					lookme(0, n2) = alld[0][r];
					lookme(1, n2) = alld[1][r];
					lookme(2, n2) = alld[3][r];
					n2++;
				}
			}*/
			

			int n1 = 0;
			int n2 = 0;

			int tu = 10000;
			int tm = 10000;
			int lookn = 100000;
			int telln = 1000;

			arma::u64_mat tellme(2, telln);

			//arma::u64_mat tellme2(2, probe_n);

			arma::mat lookme(3, lookn);
			//vec reality(telln);

			for (int u = 0; u < tu; u++) {
				for (int m = 0; m < tm; m++) {
					double r = ((double)rand() / (RAND_MAX));
					double r2 = ((double)rand() / (RAND_MAX));
					if (r < 2*lookn / ((double)(tu*tm)) && n2<lookn) {
						lookme(0, n2) = u;
						lookme(1, n2) = m;
						lookme(2, n2) = ((rand() % 10) +1);
						
						n2++;
					}
					if (r2 < 2 * telln / ((double)(tu*tm)) && n1<telln) {
						tellme(0, n1) = u;
						tellme(1, n1) = m;
						n1++;
					}

				}
			}
			cout << n1 << endl;
			cout << n2 << endl;
			//cin.get();
			cout << "ready to learn" << endl;

			size_t neighborhood = 5;
			size_t rank = 20;

			clock_t t1, t2;
			//cin.get();
			
			
			
			//CFType<> cf(lookme);
			t1 = clock();
			CFType<NMFPolicy> cf(lookme, NMFPolicy(), 1, 100, 50, 1e-100, false);//rank,iterations
			t2 = clock();
			cout << "svd\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;																 //t1 = clock();
			//CFType<cf::RegSVDPolicy> cf2(lookme, RegSVDPolicy(80), 5, 30, 80, 1e-5, false);
			//t2 = clock();
			
			//CFType<cf::BiasSVDPolicy> cf3(lookme, BiasSVDPolicy(5,0.02,0.05,&lookme), 5, 1, 1, 1e-5, false);
			cout << "NMFr" << endl;

			



			//CFType<cf::SVDPlusPlusPolicy> cf(lookme);


			std::cout << "model trained" << std::endl;
			std::cout << "in sample test start" << std::endl;
			/*
			for (int w = 0; w < 5; w++) {
				std::cout << "trial\t" <<w<< std::endl;
				std::cout << lookme(2, w) << std::endl;
				std::cout << cf.Predict(lookme(0, w), lookme(1, w)) << std::endl;
				std::cout << cf2.Predict(lookme(0, w), lookme(1, w)) << std::endl;
				std::cout << cf3.Predict(lookme(0, w), lookme(1, w)) << std::endl;
			}*/

			std::cout << "in sample test done" << std::endl;
			//cout << "!";
			//cout << cf.Predict<cf::PearsonSearch>(tellme(0, 0), tellme(1, 0)) << endl;
			
			vec predictions1(telln);
			vec predictions2(telln);
			//vec predictions2(probe_n);
			t1 = clock();
			cf.Predict(tellme, predictions1);
			t2= clock();
			double sh = (t2 - t1) / (double)CLOCKS_PER_SEC;
			
			t1 = clock();
			//cf2.Predict(tellme, predictions2);
			t2 = clock();
			cout << "svd\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;

			std::cout << "model evaluated" << std::endl;
			string outfile1 = "C:/Users/DFCTech/Desktop/out_qual_nmf_t.dta";
			string outfile2 = "C:/Users/DFCTech/Desktop/out_rs_nmf_t.dta";
			tof(outfile1, predictions1);
			tof(outfile2, predictions2);
			std::cout << "results exported" << std::endl;
			cout << "knn\t" << sh << endl;

			cin.get();

			break;
	}
	//NMF with various normalization policies
	case 9:
	{
		
		int r_alln = 102416306;
		string test1 = "C:/Users/DFCTech/Desktop/testo.dta";
		string test2 = "C:/Users/DFCTech/Desktop/testo.idx";

		string real1 = "C:/Users/DFCTech/Downloads/um/all.dta";
		string real2 = "C:/Users/DFCTech/Downloads/um/all.idx";

		int alln = (int)(r_alln);

		vector <vector <int>> alld = fromf2(real1, r_alln, 4);//colum row
		cout << "file 1 read" << endl;
		vector <vector <int>> idxd = fromf2(real2, r_alln, 1);
		cout << "file 2 read" << endl;
		/*
		vector<int> ri(r_alln);
		for (int i = 0; i < r_alln; i++)
		{
		ri[i] = i;
		}


		random_shuffle(ri.begin(), ri.end());

		vector<vector<int>> alld(4);
		for (int i = 0; i < 4; i++)
		{
		alld[i] = vector<int>(alln, 0);
		}
		vector<vector<int>> idxd(1);
		for (int i = 0; i < 1; i++)
		{
		idxd[i] = vector<int>(alln, 0);
		}

		for (int i = 0; i < alln; i++) {
		for (int j = 0; j < 4; j++)
		{
		alld[j][i] = r_alld[j][ri[i]];
		}
		for (int j = 0; j < 1; j++)
		{
		idxd[j][i] = r_idxd[j][ri[i]];
		}

		}
		*/
		
		int probe_n = 0;
		int train_n = 0;
		int qual_n = 0;

		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				probe_n++;
			}
		if (idxd[0][r] == 5) {
			qual_n++;
		}
		else {
			train_n++;
		}
		}

		
		arma::u64_mat tellme(2, qual_n);

		arma::u64_mat tellme2(2, probe_n);

		arma::mat lookme(3, train_n);

		int n1 = 0;
		int n2 = 0;
		int n3 = 0;

		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				tellme2(0, n3) = alld[0][r];
				tellme2(1, n3) = alld[1][r];
				n3++;
			}
			if (idxd[0][r] == 5) {
				tellme(0, n1) = alld[0][r];
				tellme(1, n1) = alld[1][r];
				n1++;
			}
			else {
				lookme(0, n2) = alld[0][r];
				lookme(1, n2) = alld[1][r];
				lookme(2, n2) = alld[3][r];
				n2++;
			}
		}

		/*
		int n1 = 0;
		int n2 = 0;

		int tu = 1000;
		int tm = 1000;
		int lookn = 100000;
		int telln = 100;

		arma::u64_mat tellme(2, telln);

		//arma::u64_mat tellme2(2, probe_n);

		arma::mat lookme(3, lookn);
		//vec reality(telln);

		for (int u = 0; u < tu; u++) {
			for (int m = 0; m < tm; m++) {
				double r = ((double)rand() / (RAND_MAX));
				double r2 = ((double)rand() / (RAND_MAX));
				if (r < 2 * lookn / ((double)(tu*tm)) && n2<lookn) {
					lookme(0, n2) = u;
					lookme(1, n2) = m;
					lookme(2, n2) = (rand() % 5 + 1);

					n2++;
				}
				if (r2 < 2 * telln / ((double)(tu*tm)) && n1<telln) {
					tellme(0, n1) = u;
					tellme(1, n1) = m;
					n1++;
				}

			}
		}*/
		cout << n1 << endl;
		cout << n2 << endl;
		//cin.get();
		cout << "ready to learn" << endl;

		size_t neighborhood = 5;
		size_t rank = 20;

		clock_t t1, t2;
		//cin.get();

		t1 = clock();

		//CFType<> cf(lookme);
		CFType<NMFPolicy, OverallMeanNormalization> cf(lookme, NMFPolicy(), 1, 5, 30, 1e-5, false);//rank,iterations
		cout << "NMF" << endl;




		//CFType<cf::SVDPlusPlusPolicy> cf(lookme);
		t2 = clock();

		cout << "train time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;


		std::cout << "model trained" << std::endl;

		//cout << "!";
		//cout << cf.Predict<cf::PearsonSearch>(tellme(0, 0), tellme(1, 0)) << endl;

		vec predictions1(qual_n);
		vec predictions2(probe_n);
		t1 = clock();
		cf.Predict(tellme, predictions1);
		cf.Predict(tellme2, predictions2);
		t2 = clock();
		cout << "predict time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;

		std::cout << "model evaluated" << std::endl;
		string outfile1 = "C:/Users/DFCTech/Desktop/out_qual_nmf.dta";
		string outfile2 = "C:/Users/DFCTech/Desktop/out_probe_nmf.dta";
		tof(outfile1, predictions1);
		tof(outfile2, predictions2);
		std::cout << "results exported" << std::endl;


		cin.get();

		break;
	}
	//reg svd with certain parameters and pearson distance metric
	case 8: {
		int alln = 102416306;
		alln = (int)(alln);
		string test1 = "C:/Users/DFCTech/Desktop/testo.dta";
		string test2 = "C:/Users/DFCTech/Desktop/testo.idx";

		string real1 = "C:/Users/DFCTech/Downloads/um/all.dta";
		string real2 = "C:/Users/DFCTech/Downloads/um/all.idx";


		vector <vector <int>> alld = fromf2(real1, alln, 4);//colum row
		cout << "file 1 read" << endl;
		vector <vector <int>> idxd = fromf2(real2, alln, 1);
		cout << "file 2 read" << endl;

		int probe_n = 0;
		int train_n = 0;
		int qual_n = 0;

		for (int r = 0; r < alln; r++) {
				if (idxd[0][r] == 4) {
					probe_n++;
				}
				else if (idxd[0][r] == 5) {
					qual_n++;
				}
				else {
					train_n++;
				}
		}


		arma::u64_mat tellme(2, qual_n);

		arma::u64_mat tellme2(2, probe_n);

		arma::mat lookme(3, train_n);

		int n1 = 0;
		int n2 = 0;
		int n3 = 0;

		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				tellme2(0, n3) = alld[0][r];
				tellme2(1, n3) = alld[1][r];
				n3++;
			}
			else if (idxd[0][r] == 5) {
				tellme(0, n1) = alld[0][r];
				tellme(1, n1) = alld[1][r];
				n1++;
			}
			else {
				lookme(0, n2) = alld[0][r];
				lookme(1, n2) = alld[1][r];
				lookme(2, n2) = alld[3][r];
				n2++;
			}
		}

		cout << "ready to learn" << endl;

		size_t neighborhood = 5;
		size_t rank = 20;

		clock_t t1, t2;

		t1 = clock();

		CFType<cf::RegSVDPolicy> cf(lookme, RegSVDPolicy(80, 0.015, 0.04), 7, 40, 80, 1e-5, false);
		//CFType<cf::BiasSVDPolicy> cf(lookme, BiasSVDPolicy(1, 0.02, 0.05, &lookme), 5, 1, 1, 1000, false);
		cout << "svd s" << endl;




		//CFType<cf::SVDPlusPlusPolicy> cf(lookme);
		t2 = clock();

		cout << "train time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;


		std::cout << "model trained" << std::endl;

		//cout << "!";
		//cout << cf.Predict<cf::PearsonSearch>(tellme(0, 0), tellme(1, 0)) << endl;

		vec predictions1(qual_n);
		vec predictions2(probe_n);
		t1 = clock();
		cf.Predict<PearsonSearch>(tellme, predictions1);
		cf.Predict<PearsonSearch>(tellme2, predictions2);
		t2 = clock();
		cout << "predict time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;

		std::cout << "model evaluated" << std::endl;
		string outfile1 = "C:/Users/DFCTech/Desktop/out_qual_knn_2.dta";
		string outfile2 = "C:/Users/DFCTech/Desktop/out_probe_knn_2.dta";
		tof(outfile1, predictions1);
		tof(outfile2, predictions2);
		std::cout << "results exported" << std::endl;
		cin.get();

		break; 
}
	//regsvd with pearson search, also one probe, and parameters to compare another svd implementation
	case 7: {
		int alln = 102416306;
		alln = (int)(alln);
		string test1 = "C:/Users/DFCTech/Desktop/testo.dta";
		string test2 = "C:/Users/DFCTech/Desktop/testo.idx";

		string real1 = "C:/Users/DFCTech/Downloads/um/all.dta";
		string real2 = "C:/Users/DFCTech/Downloads/um/all.idx";


		vector <vector <int>> alld = fromf2(real1, alln, 4);//colum row
		cout << "file 1 read" << endl;
		vector <vector <int>> idxd = fromf2(real2, alln, 1);
		cout << "file 2 read" << endl;

		int probe_n = 0;
		int train_n = 0;
		int qual_n = 0;

		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				probe_n++;
			}
			if (idxd[0][r] == 5) {
				qual_n++;
			}
			else {
				train_n++;
			}
		}

		arma::u64_mat tellme(2, qual_n);

		arma::u64_mat tellme2(2, probe_n);

		arma::mat lookme(3, train_n);

		int n1 = 0;
		int n2 = 0;
		int n3 = 0;

		for (int r = 0; r < alln; r++) {
			if (idxd[0][r] == 4) {
				tellme2(0, n3) = alld[0][r];
				tellme2(1, n3) = alld[1][r];
				n3++;
			}
			if (idxd[0][r] == 5) {
				tellme(0, n1) = alld[0][r];
				tellme(1, n1) = alld[1][r];
				n1++;
			}
			else {
				lookme(0, n2) = alld[0][r];
				lookme(1, n2) = alld[1][r];
				lookme(2, n2) = alld[3][r];
				n2++;
			}
		}

		cout << "ready to learn" << endl;

		size_t neighborhood = 5;
		size_t rank = 20;

		clock_t t1, t2;

		t1 = clock();

		CFType<cf::RegSVDPolicy> cf(lookme, RegSVDPolicy(80), 5, 30, 80, 1e-5, false);
		cout << "svd s" << endl;




		//CFType<cf::SVDPlusPlusPolicy> cf(lookme);
		t2 = clock();

		cout << "train time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;


		std::cout << "model trained" << std::endl;

		//cout << "!";
		//cout << cf.Predict<cf::PearsonSearch>(tellme(0, 0), tellme(1, 0)) << endl;

		vec predictions1(qual_n);
		vec predictions2(probe_n);
		t1 = clock();
		cf.Predict<PearsonSearch>(tellme, predictions1);
		cf.Predict<PearsonSearch>(tellme2, predictions2);
		t2 = clock();
		cout << "predict time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;

		std::cout << "model evaluated" << std::endl;
		string outfile1 = "C:/Users/DFCTech/Desktop/out_qual_r_svd_2.dta";
		string outfile2 = "C:/Users/DFCTech/Desktop/out_probe_r_svd_2.dta";
		tof(outfile1, predictions1);
		tof(outfile2, predictions2);
		std::cout << "results exported" << std::endl;
		cin.get();

		break;
	}
	//execute certain methods for blending in succesion
	case 10: {
		
		/*
		doknn(true, "C:/Users/DFCTech/Desktop/f_knn_p.dta");
		doknn(false, "C:/Users/DFCTech/Desktop/f_knn_q.dta");
		dosvd(true,  "C:/Users/DFCTech/Desktop/f_rsvd_p.dta");
		dosvd(false, "C:/Users/DFCTech/Desktop/f_rsvd_q.dta");
		*/

		/*
		doknn2svd(true, "C:/Users/DFCTech/Desktop/f_knn2svd_p_4.dta");
		doknn2svd(false, "C:/Users/DFCTech/Desktop/f_knns2svd_q_4.dta");
		*/

		doknn2(true, "C:/Users/DFCTech/Desktop/f_knn2_p_4.dta");
		doknn2(false, "C:/Users/DFCTech/Desktop/f_knn2_q_4.dta");
		cin.get();
		
		break;
	}
	//knn and other neighborhood method testing on simulated data, used for analysis with mathematica
		case 11:
		{
			int n1 = 0;
			int n2 = 0;

			int tu = 1000;
			int tm = 1000;
			int lookn = 100000;
			int telln = 1000000;

			arma::mat tellme(3, telln);

			//arma::u64_mat tellme2(2, probe_n);

			arma::mat lookme(3, lookn);
			vec ratings(lookn);
			//vec reality(telln);

			for (int u = 0; u < tu; u++) {
				for (int m = 0; m < tm; m++) {
					double r = ((double)rand() / (RAND_MAX));
					double r2 = ((double)rand() / (RAND_MAX));
					if (r < 2 * lookn / ((double)(tu*tm)) && n2<lookn) {
						lookme(0, n2) = u;
						lookme(1, n2) = m;
						lookme(2, n2) = u+m;
						ratings(n2) = ((rand() % 5) + 1);

						n2++;
					}
					if (r2 < 2 * telln / ((double)(tu*tm)) && n1<telln) {
						tellme(0, n1) = u;
						tellme(1, n1) = m;
						tellme(2, n1) = u+m;
						n1++;
					}

				}
			}

			//NeighborSearch<> a(lookme,DUAL_TREE_MODE,0.5);
			
			//a.Epsilon() = .5;
			//a.Train(lookme);
			//PearsonSearch

			arma::mat tellme_N;
			tellme_N = arma::normalise(tellme.each_row() - arma::mean(tellme));

			arma::mat lookme_N(arma::size(lookme));
			lookme_N = arma::normalise(
				lookme.each_row() - arma::mean(lookme));

			Mat<size_t> neighborhood;
			mat similarities;
			clock_t t1, t2;
			t1 = clock();
			
			RASearch<> a(lookme_N, true,true,5,0.75,false,false,20);
			//RASearch<> a(lookme_N);
			//NeighborSearch<> a(lookme_N, GREEDY_SINGLE_TREE_MODE, 0.05);
			//KNN a(lookme_N);
			a.Search(tellme_N, 5, neighborhood, similarities);
			t2 = clock();
			cout << "predict time\t" << (t2 - t1) / (double)CLOCKS_PER_SEC << endl;
			//cout << similarities << endl;
			cout << neighborhood << endl;
			cin.get();

			vec predictions(tellme_N.n_cols);

			for (int u = 0; u < neighborhood.n_cols; u++) {
				double rating = 0;
				for (int n = 0; n < neighborhood.n_rows; n++) {
					rating += ratings(neighborhood(n, u));
				}
				predictions(u) = rating / neighborhood.n_rows;
			}

			break;
		}
	}
}



//std::vector<std::vector<int>> data(102416306, std::vector<int>(4));	
//cout << "omfg!" << endl;

/*
int rs = (int)(alln / 100);
int** alld = fromf3(real1, rs, 4);
cout << "file2" << endl;
int** idxd = fromf3(real2, rs, 1);
std::cout << "data imported" << std::endl;
cout << alld[1][1]<<endl;
std::getchar();
int num5 = 0;
for (int i = 0; i < rs; i++) {
if (idxd[i][0] == 5) {
num5++;
}
}
*/


//int numelse = rs - num5;



//0 user, 1 movie, 2 date, 3 rating
/*
for (int r = 0; r < alld.size(); r++) {
if (idxd[r][0] == 5) {
tellme(0, n1) = alld[r][0];
tellme(1, n1) = alld[r][1];
n1++;
}
else {
lookme(0, n2) = alld[r][0];
lookme(1, n2) = alld[r][1];
lookme(2, n2) = alld[r][3];
n2++;
}
}

std::cout << lookme(0,0) << std::endl;
std::cout << lookme(0,1) << std::endl;
*/


//int rows = sizeof alld / sizeof alld[0]; // 2 rows  

//int cols = sizeof alld[0] / sizeof(int); // 5 cols




//std::getchar();


//arma::mat d; // (user, item, rating) table
//data::Load("C:/Users/DFCTech/Downloads/mlpack-3.0.4/src/mlpack/tests/data/german2.csv", d, true);
//arma::Col<size_t> users; // users seeking recommendations
//data::Load("C:/Users/DFCTech/Downloads/mlpack-3.0.4/src/mlpack/tests/data/german2.csv", d, true);
