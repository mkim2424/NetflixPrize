#include "util.h"

void print(std::string str) {
    std::cout << str << std::flush;
}

void println(std::string str) {
    std::cout << str << std::endl;
}

void print(std::string str, int val) {
    std::cout << str << " = " << val << std::endl;
}

void print(std::string str, double val) {
    std::cout << str << " = " << val << std::endl;
}

void print_parameters(Parameter p) {
    println("Parameters:");
    
    switch (p.mt) {
        case Baseline_t:
            println("Model type = Baseline");
            break;
        case Time_baseline_t:
            println("Model type = Time_baseline");
            break;
        case Time_SVD_t:
            println("Model type = Time_SVD");
            break;
        case SVD_t:
            println("Model type = SVD");
            break;
        case SVDpp_t:
            println("Model type = SVDpp");
            break;
        case Time_SVDpp_t:
            println("Model type = Time_SVDpp");
            break;
        case FactorNM_t:
            println("Model type = FactorNM");
            break;
    }

    print("k", p.k);
    print("maxEpoch", p.maxEpoch);
    println("Regularizers");

    for (int i = 0; i < (int) p.reg.size(); i++) {
    	std::cout << p.reg[i] << std::endl;
    }

    println("Learning rates");

    for (int i = 0; i < (int) p.eta.size(); i++) {
    	std::cout << p.eta[i] << std::endl;
    }

    print("decay", p.decay);
    println("");
}

double clip(double value) {
    return value > 5 ? 5 : (value < 1 ? 1 : value);
}

int get_bin(int date) {
    int val = date / 70;
    return val > 31 ? 31 : val;
}

double sign(double val) {
    return val > 0 ? 1.0 : -1.0;
}
