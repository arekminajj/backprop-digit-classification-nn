#pragma once

#include <vector>
#include <Eigen/Dense>
#include <numeric>
#include <chrono>
#include <random>
#include <iostream>

using Eigen::MatrixXd;

class Network {
private:
	std::vector<int> layers;
	int numLayers;
	std::vector<Eigen::MatrixXd> biases;
	std::vector<Eigen::MatrixXd> weights;

	void backprop(const Eigen::VectorXd& image, unsigned char label,
		std::vector<Eigen::MatrixXd>& delta_nabla_b,
		std::vector<Eigen::MatrixXd>& delta_nabla_w);
	void updateMiniBatch(const std::vector<std::vector<unsigned char>>& miniBatchImages,
		const std::vector<unsigned char>& miniBatchLabels,
		double eta);
	Eigen::MatrixXd feedForward(const Eigen::VectorXd& image);

	static double sigmoid(double x)
	{
		return 1.0 / (1.0 + exp(-x));
	}

	static double applySigmoidPrime(double x)
	{
		double sig = sigmoid(x);
		return sig * (1 - sig);
	}

	Eigen::MatrixXd sigmoidPrime(const Eigen::MatrixXd& zs) const
	{
		return zs.unaryExpr(&Network::applySigmoidPrime);
	}

	Eigen::MatrixXd costDerivative(const Eigen::MatrixXd& output_activations, unsigned char label) const
	{
		Eigen::MatrixXd target = Eigen::MatrixXd::Zero(output_activations.rows(), 1);
		target(static_cast<int>(label), 0) = 1.0;
		//chcemy tutaj taka macierz ktora jest 1 dla poprawnej klasy i 0 dla innych
		//wiec dla label = 0 byloby [1, 0, 0 ....., 0]
		return output_activations - target;
	}

public:
	Network(const std::vector<int>& layers);
	void SGD(const std::vector<std::vector<unsigned char>>& trainImages,
		const std::vector<unsigned char>& trainLabels,
		int epochs, int miniBatchSize, double eta);
	int Evaluate(const std::vector<std::vector<unsigned char>>& testImages,
		const std::vector<unsigned char>& testLabels);
	void saveNetwork(const std::string& filename);
	void loadNetwork(const std::string& filename);
};