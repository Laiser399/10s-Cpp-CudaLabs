#include <iostream>
#include <fstream>
#include <tuple>
#include <utility>
#include <vector>

using std::cout;
using std::cin;
using std::endl;
using std::ifstream;
using std::istream;
using std::string;
using std::vector;
using std::tuple;


class ProgramConfiguration {
private:
    string inputFilePath, outputFilePath;
    vector<vector<tuple<int, int>>> samples;
public:
    ProgramConfiguration(
            string inputFilePath,
            string outputFilePath,
            vector<vector<tuple<int, int>>> samples
    ) {
        this->inputFilePath = std::move(inputFilePath);
        this->outputFilePath = std::move(outputFilePath);
        this->samples = std::move(samples);
    }

    const string &getInputFilePath() const {
        return inputFilePath;
    }

    const string &getOutputFilePath() const {
        return outputFilePath;
    }

    const vector<vector<tuple<int, int>>> &getSamples() const {
        return samples;
    }
};


ProgramConfiguration readProgramConfiguration(istream &inputStream) {
    string inputFilePath, outputFilePath;
    int classCount;
    vector<vector<tuple<int, int>>> samples;

    inputStream >> inputFilePath;
    inputStream >> outputFilePath;
    inputStream >> classCount;

    for (int i = 0; i < classCount; ++i) {
        int pixelCount;
        inputStream >> pixelCount;
        vector<tuple<int, int>> pixels;

        for (int j = 0; j < pixelCount; ++j) {
            int x, y;
            inputStream >> x >> y;
            pixels.emplace_back(x, y);
        }

        samples.push_back(pixels);
    }

    return {inputFilePath, outputFilePath, samples};
}


ProgramConfiguration readProgramConfiguration(int argc, char *argv[]) {
    if (argc == 2) {
        auto configurationFilePath = argv[1];

        ifstream inputFile(configurationFilePath);
        auto configuration = readProgramConfiguration(inputFile);
        inputFile.close();

        return configuration;
    }

    return readProgramConfiguration(cin);
}


int main(int argc, char *argv[]) {
    auto configuration = readProgramConfiguration(argc, argv);

    return 0;
}
