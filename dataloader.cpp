#include "dataloader.h"
#include <iostream>


dataloader::dataloader(std::string path = "./data", int batch = 1){
    batch_size = batch;

    dir = opendir(path.c_str());
    while (pdir = readdir(dir)) {
        std::string full_path = path;
        std::string file_name = pdir -> d_name;
        full_path.append("/");
        full_path.append(pdir->d_name);
        if (file_name.compare(".") != 0 && file_name.compare("..") != 0){
            files.push_back(full_path);
        }
    }
    shuffle();
    iter = begin();
}
void dataloader::shuffle(){
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(files.begin(), files.end(), g);
}
dataloader::iterator dataloader::begin(){
    return files.begin();
}
dataloader::iterator dataloader::end(){
    return files.end();
}


bool dataloader::get_batch(matrix &batch, matrix &target){

    float tags[batch_size];
    for (int b = 0; b < batch_size; b++){
        if (iter == end()){
            shuffle();
            iter = begin();
            return false;
        }

        std::string name = *iter++;
        std::string stag = name.substr(name.find("_") - 1,  1);
        tags[b] = atof(stag.c_str());

        cv::Mat img = cv::imread(name);
        img.convertTo(img, CV_32F);
        img /= 255.0;

 
        matrix tmp(1, img.rows*img.cols, (float*)img.data, true);
        if (batch.height == -1){
            batch = tmp;
        }
        else{
            batch = cat(batch, tmp, 0);
        }
    }
    matrix tmp(batch_size, 1, tags, true);
    target = tmp;
    return true;
}
