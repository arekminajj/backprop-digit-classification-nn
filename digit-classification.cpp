#include "DataManager.h"
#include "Network.h"

int main()
{
	std::string trainImagesFile = "train-images.idx3-ubyte";
	std::string trainLabelsFile = "train-labels.idx1-ubyte";

	std::string testImagesFile = "t10k-images.idx3-ubyte";
	std::string testLabelsFile = "t10k-labels.idx1-ubyte";

	DataManager manager;

	std::vector<std::vector<unsigned char>> train_images = manager.loadMnistImages(trainImagesFile);
	std::vector<unsigned char> train_labels = manager.loadMnistLabels(trainLabelsFile);

	std::cout << "Loaded " << train_images.size() << " training images and "
		<< train_labels.size() << " training labels.\n";


	std::vector<std::vector<unsigned char>> test_images = manager.loadMnistImages(testImagesFile);
	std::vector<unsigned char> test_labels = manager.loadMnistLabels(testLabelsFile);

	std::cout << "Loaded " << test_images.size() << " test images and "
		<< test_labels.size() << " test labels.\n";


	std::vector<int> layers = { 784, 128, 64, 10 };
	Network net(layers);

	net.SGD(train_images, train_labels, 30, 32, 0.05);
	std::cout << "trained";
	//net.loadNetwork("backpropNetwork.bin");
	net.Evaluate(test_images, test_labels);

	return 0;
}