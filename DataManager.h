#pragma once

#include <vector>
#include <fstream>
#include <iostream>

class DataManager
{
private:
	int readInt(std::ifstream& file);

public:
	std::vector<std::vector<unsigned char>> loadMnistImages(const std::string& filename);
	std::vector<unsigned char> loadMnistLabels(const std::string& filename);
};

