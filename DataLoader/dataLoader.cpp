#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "dataLoader.h"
#include "../Matrix/Matrix.h"

using namespace std;

int readInt(ifstream &file)
{
    char buffer[4];
    file.read(buffer, 4);
    return __builtin_bswap32(*reinterpret_cast<int *>(buffer)); // EMNISTT file stored in big endian. Mac read lil endian. So we convert big to lil endian
}

vector<int> loadLabels(const string &filename)
{
    ifstream fileRead(filename, ios::binary);
    if (!fileRead.is_open())
        throw runtime_error("Could not open file: " + filename);

    int magicNo = readInt(fileRead);
    if (magicNo != 2049)
        throw runtime_error("File has invalid labels");

    int labelNo = readInt(fileRead);
    vector<int> labels(labelNo);

    for (int i = 0; i < labelNo; i++)
    {
        unsigned char label;                                // each label stored as 1 byte in EMNIST file and unsigned char is the natural cpp type for 1 raw byte
        fileRead.read(reinterpret_cast<char *>(&label), 1); //.read expects first argument to be char*
        labels[i] = (static_cast<int>(label));
    }

    fileRead.close();
    return labels;
}

vector<Matrix> loadImages(const string &filename)
{
    ifstream fileRead(filename, ios::binary);

    if (!fileRead.is_open())
        throw runtime_error("Could not open file: " + filename);

    int magic = readInt(fileRead);
    if (magic != 2051)
        throw runtime_error("Invalid images file");

    int imageNo = readInt(fileRead);
    int rowNo = readInt(fileRead);
    int colNo = readInt(fileRead);
    int pixels = rowNo * colNo; // should be 784

    vector<Matrix> images;
    images.reserve(imageNo);

    for (int i = 0; i < imageNo; i++)
    {
        // read raw bytes
        vector<unsigned char> buffer(pixels);
        fileRead.read(reinterpret_cast<char *>(buffer.data()), pixels);

        Matrix img(pixels, 1);
        for (int j = 0; j < pixels; j++)
            img(j, 0) = static_cast<double>(buffer[j]) / 255.0; // values in data file b/w 0-255, normalise them to 0-1

        images.push_back(img);
    }

    fileRead.close();
    return images;
}

std::vector<Sample> makeSamples(const std::vector<Matrix> &images, const std::vector<int> &labels)
{
    vector<Sample> samples;
    samples.reserve(images.size());
    for (int i = 0; i < images.size(); i++)
        samples.push_back({images[i], labels[i]});
    return samples;
}
