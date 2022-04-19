#include <iostream>
#include <fstream>
#include <tuple>
#include <utility>
#include <vector>

using std::cout;
using std::cin;
using std::cerr;
using std::endl;
using std::ifstream;
using std::istream;
using std::string;
using std::vector;
using std::tuple;
using std::ios_base;


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


struct Size2D {
    int width, height;

    int getSize() const {
        return width * height;
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
        if (!inputFile.is_open()) {
            cerr << "Could not open configuration file \"" << configurationFilePath << "\"" << endl;
            exit(1);
        }

        auto configuration = readProgramConfiguration(inputFile);
        inputFile.close();

        return configuration;
    }

    return readProgramConfiguration(cin);
}


tuple<Size2D, uchar4*> readInputData(const ProgramConfiguration &configuration) {
    ifstream input(configuration.getInputFilePath(), ios_base::binary);

    if (!input.is_open()) {
        cerr << "Could not open input file." << endl;
        exit(1);
    }

    Size2D sourceSize{};
    input.read((char *) &sourceSize.width, sizeof(sourceSize.width));
    input.read((char *) &sourceSize.height, sizeof(sourceSize.height));

    auto *data = new uchar4[sourceSize.getSize()];
    input.read((char *) data, (long long) sizeof(data[0]) * sourceSize.getSize());

    input.close();
}


int main(int argc, char *argv[]) {
    auto configuration = readProgramConfiguration(argc, argv);

    Size2D dataSize{};
    uchar4* data;
    std::tie(dataSize, data) = readInputData(configuration);



    return 0;
}
