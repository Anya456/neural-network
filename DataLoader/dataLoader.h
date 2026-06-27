#include <iostream>
#include <vector>
#include <string>
#include "../Matrix/Matrix.h"

struct Sample
{
    Matrix image;
    int label;
};

std::vector<Matrix> loadImages(const std::string filename);
std::vector<int> loadLabels(const std::string filename);
std::vector<Sample> makeSamples(const std::vector<Matrix> &images, const std::vector<int> &labels);
