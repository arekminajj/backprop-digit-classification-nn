#include "Network.h"

void Network::backprop(
    const Eigen::VectorXd& image,
    unsigned char label,
    std::vector<Eigen::MatrixXd>& delta_nabla_b,
    std::vector<Eigen::MatrixXd>& delta_nabla_w)
{
    std::vector<Eigen::MatrixXd> activations(numLayers);
    std::vector<Eigen::MatrixXd> zs(numLayers - 1);

    //warstwa wejsciowa
    activations[0] = image;

    for (int i = 0; i < numLayers - 1; i++) {
        zs[i] = weights[i] * activations[i] + biases[i];
        activations[i + 1] = zs[i].unaryExpr(&Network::sigmoid);
    }

    auto costFunctionDerivative = costDerivative(activations.back(), label);
    Eigen::MatrixXd delta = costFunctionDerivative.cwiseProduct(sigmoidPrime(zs.back()));

    delta_nabla_b.back() = delta;
    delta_nabla_w.back() = delta * activations[numLayers - 2].transpose();

    for (int l = 2; l < numLayers; ++l) {
        int index = numLayers - l;

        delta = (weights[index].transpose() * delta).cwiseProduct(sigmoidPrime(zs[index - 1]));

        delta_nabla_b[index - 1] = delta;
        delta_nabla_w[index - 1] = delta * activations[index - 1].transpose();
    }
}

void Network::updateMiniBatch(
    const std::vector<std::vector<unsigned char>>& miniBatchImages,
    const std::vector<unsigned char>& miniBatchLabels,
    double eta)
{
    int n = miniBatchImages.size();
    std::vector<Eigen::MatrixXd> nabla_b(biases.size());
    std::vector<Eigen::MatrixXd> nabla_w(weights.size());

    for (size_t i = 0; i < nabla_b.size(); i++) {
        nabla_b[i] = Eigen::MatrixXd::Zero(biases[i].rows(), biases[i].cols());
        nabla_w[i] = Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols());
    }

    std::vector<Eigen::VectorXd> converted_images(n);

    for (int i = 0; i < n; i++) {
        std::vector<double> image(miniBatchImages[i].begin(), miniBatchImages[i].end());
        converted_images[i] = Eigen::Map<Eigen::VectorXd>(image.data(), image.size());
    }

    for (int i = 0; i < n; i++) {
        std::vector<Eigen::MatrixXd> delta_nabla_b(biases.size());
        std::vector<Eigen::MatrixXd> delta_nabla_w(weights.size());

        for (size_t j = 0; j < delta_nabla_b.size(); j++) {
            delta_nabla_b[j] = Eigen::MatrixXd::Zero(biases[j].rows(), biases[j].cols());
            delta_nabla_w[j] = Eigen::MatrixXd::Zero(weights[j].rows(), weights[j].cols());
        }

        Eigen::VectorXd image = converted_images[i];

        backprop(image, miniBatchLabels[i], delta_nabla_b, delta_nabla_w);

        for (size_t j = 0; j < nabla_b.size(); j++) {
            nabla_b[j] += delta_nabla_b[j];
            nabla_w[j] += delta_nabla_w[j];
        }
    }

    for (size_t i = 0; i < biases.size(); i++) {
        biases[i] -= (eta / n) * nabla_b[i];
        weights[i] -= (eta / n) * nabla_w[i];
    }
}

Eigen::MatrixXd Network::feedForward(const Eigen::VectorXd& image)
{
    Eigen::VectorXd activation = image;

    for (int i = 0; i < numLayers - 1; i++) {
        activation = (weights[i] * activation + biases[i]).unaryExpr(&Network::sigmoid);
    }

    return activation;
}

Network::Network(const std::vector<int>& layers)
	: layers(layers), numLayers(layers.size())
{
    biases.resize(numLayers - 1);
    weights.resize(numLayers - 1);

    for (int i = 0; i < numLayers - 1; i++) {
        biases[i] = MatrixXd::Zero(layers[i + 1], 1);
        weights[i] = MatrixXd::Random(layers[i + 1], layers[i]);
    }
}

void Network::SGD(const std::vector<std::vector<unsigned char>>& trainImages, const std::vector<unsigned char>& trainLabels, int epochs, int miniBatchSize, double eta)
{
    int n = trainImages.size();
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());

    for (int epoch = 0; epoch < epochs; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        std::shuffle(indices.begin(), indices.end(), g);

        for (int j = 0; j < n; j += miniBatchSize) {
            std::vector<std::vector<unsigned char>> miniBatchImages;
            std::vector<unsigned char> miniBatchLabels;

            for (int k = j; k < std::min(j + miniBatchSize, n); ++k) {
                miniBatchImages.push_back(trainImages[indices[k]]);
                miniBatchLabels.push_back(trainLabels[indices[k]]);
            }

            auto start = std::chrono::high_resolution_clock::now();
            updateMiniBatch(miniBatchImages, miniBatchLabels, eta);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            //std::cout << "Mini-batch processed in " << elapsed.count() << " seconds." << std::endl;
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;
        std::cout << "Epoch " << epoch << " complete in " << epoch_duration.count() << " seconds.\n";
        Evaluate(trainImages, trainLabels);
    }
}

int Network::Evaluate(const std::vector<std::vector<unsigned char>>& testImages, const std::vector<unsigned char>& testLabels)
{
    int numCorrect = 0;
    int n = testImages.size();

    for (size_t i = 0; i < n; i++) {
        std::vector<double> image(testImages[i].begin(), testImages[i].end());
        Eigen::VectorXd vectorImage = Eigen::Map<Eigen::VectorXd>(image.data(), image.size());

        Eigen::VectorXd output = feedForward(vectorImage);

        int predicted_class;
        output.maxCoeff(&predicted_class);

        if (predicted_class == static_cast<int>(testLabels[i])) {
            numCorrect++;
        }
    }

    double accuracy = static_cast<double>(numCorrect) / n;
    std::cout << "Accuracy: " << accuracy * 100.0 << "%\n";
    return numCorrect;
}
