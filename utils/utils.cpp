#include "utils.h"

void readBinData(PinnedFloatVector& re, PinnedFloatVector& im, std::string fileName)
{
    std::ifstream signalFile;
    signalFile.open(fileName, std::ios::binary);

    if (!signalFile)
    {
        std::cout << " Error, Couldn't find the input file" << "\n";
        exit(0);
    }
    int num_elements{};
    signalFile.seekg(0, std::ios::end);
    num_elements = signalFile.tellg() / sizeof(float) / 2;
    signalFile.seekg(0, std::ios::beg);
    re.resize(num_elements);
    im.resize(num_elements);
    std::cout << "number of samples in the file: " << num_elements << std::endl;

    std::string signalLine;
    int i{};
    float f;

    while ((!signalFile.eof()) && (i < num_elements))
    {
        signalFile.read(reinterpret_cast<char*>(&f), sizeof(float));
        re[i] = f;
        signalFile.read(reinterpret_cast<char*>(&f), sizeof(float));
        im[i] = f;
        i++;
    }
    std::cout << "number of read samples: " << i << std::endl;
    signalFile.close();
}

void recordData(float *data_re, float *data_im, int sizeOfWrite, std::string fileName)
{
    std::ofstream outFile;
    outFile.open(fileName, std::ios::binary);
    if (outFile.is_open()) {
        std::cout << "isOpen, writing " << sizeOfWrite << " symbols to "<<fileName<<"(float32)"<< std::endl;
    }
    else
        std::cout << "isNotOpen" << std::endl;

    for (int i = 0; i < sizeOfWrite; i++)
    {
        outFile.write(reinterpret_cast<const char*>(&data_re[i]), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&data_im[i]), sizeof(float));
    }
    outFile.close();
}

