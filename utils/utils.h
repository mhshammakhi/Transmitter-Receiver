#pragma once

#ifndef UTILS_H
#define UTILS_H

#include "pinnedMemory.hpp"

#include<iostream>
#include<string>
#include<fstream>
#include<vector>

void readBinData(PinnedFloatVector& re, PinnedFloatVector& im, std::string fileName);
void recordData(float *data_re, float *data_im, int sizeOfWrite, std::string fileName);

#endif // UTILS_H
