#include <iostream>
#include <tuple>
#include <string>
#include <fstream>

using std::cout;
using std::cerr;
using std::cin;
using std::endl;
using std::istream;
using std::ifstream;
using std::tuple;
using std::string;


tuple<int, float *> readInput(istream &input) {
    int n;
    input >> n;

    auto *matrix = new float[n * n];

    for (int i = 0; i < n * n; ++i) {
        input >> matrix[i];
    }

    return tuple<int, float *>{n, matrix};
}


tuple<int, float *> readInput(int argc, char *argv[]) {
    if (argc == 2) {
        auto inputFilePath = argv[1];

        ifstream input(inputFilePath);
        if (!input.is_open()) {
            cerr << "Could not open input file \"" << inputFilePath << "\"." << endl;
            exit(1);
        }

        auto result = readInput(input);

        input.close();

        return result;
    } else {
        return readInput(cin);
    }
}


int main(int argc, char *argv[]) {
    int matrixSize;
    float *matrix;
    std::tie(matrixSize, matrix) = readInput(argc, argv);

    return 0;
}
