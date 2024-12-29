#include "DataManager.h"

// [offset] [type] [value] [description]
// 0000     32 bit integer  0x00000803(2051) magic number
// 0004     32 bit integer  60000            number of images
// 0008     32 bit integer  28               number of rows
// 0012     32 bit integer  28               number of columns
// 0016     unsigned byte ? ? pixel
// 0017     unsigned byte ? ? pixel
// ........
// xxxx     unsigned byte ? ? pixel
// Pixels are organized row - wise.Pixel values are 0 to 255. 0 means background(white), 255 means foreground(black).

// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
// 0004     32 bit integer  60000            number of items
// 0008     unsigned byte   ??               label
// 0009     unsigned byte ? ? label
// ........
// xxxx     unsigned byte ? ? label
// The labels values are 0 to 9.

int DataManager::readInt(std::ifstream& file)
{
	unsigned char buffer[4];
	file.read(reinterpret_cast<char*>(buffer), 4);
	// big-endian format
	// https://en.wikipedia.org/wiki/Endianness
	// wiec buffer[0] -> msb
	// chodzi o to aby zrobic z 4 bajtow inta
	return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
}

std::vector<std::vector<unsigned char>> DataManager::loadMnistImages(const std::string& filename)
{
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		std::cout << "Could not open file: " << filename << "\n";
	}

	int magicNumber = readInt(file);
	if (magicNumber != 2051) {
		std::cout << "Invalid magic number: " << magicNumber << " should be 2051!\n";
	}

	//pierwsze cztery bajty to magic number, kolejne 4 to liczba obrazkow, kolejne 4 to liczba wierszy, kolejne 4 to liczba kolumn
	int numImages = readInt(file);
	int numRows = readInt(file);
	int numColumns = readInt(file);

	//tworzymy wektory na obrazki
	//numImages liczbe wektorow, kazdy zawiera wektory(28*28) na kazdy pixel
	std::vector<std::vector<unsigned char>> images(numImages, std::vector<unsigned char>(numRows * numColumns));

	//dla kazdego obrazka czytamy 28*28 bajtow, kazdy na pixel i wrzucamy na wskaznik images[i]
	for (int i = 0; i < numImages; i++) {
		file.read(reinterpret_cast<char*>(images[i].data()), numRows * numColumns);
	}

	return images;
}

std::vector<unsigned char> DataManager::loadMnistLabels(const std::string& filename)
{
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		std::cout << "Could not open file: " << filename << "\n";
	}

	//ideantycznie najpierw magic number zeby zobaczyc czy plik poprawny i my dobrze czytamy go
	int magicNumber = readInt(file);
	if (magicNumber != 2049) {
		std::cout << "Invalid magic number: " << magicNumber << " should be 2049!\n";
	}

	int numLabels = readInt(file);

	//w takiej samej kolejnosci jak obrazki, label dla kazdego obrazka
	std::vector<unsigned char> labels(numLabels);

	file.read(reinterpret_cast<char*>(labels.data()), numLabels);

	return labels;
}
