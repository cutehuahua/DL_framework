#ifndef LOADER_H
#define LOADER_H
#include <dirent.h>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

#include "opencv2/opencv.hpp"
#include "core.h"

class dataloader{
public:
    int batch_size;
    DIR* dir;
    dirent* pdir;
    std::vector<std::string> files;

    dataloader(std::string path, int batch_size);
    void shuffle();
    typedef std::vector<std::string>::iterator iterator;
    iterator begin();
    iterator end();
    iterator iter;
    bool get_batch(matrix &batch, matrix &target);

};


#endif