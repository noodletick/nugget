#pragma once
#include <iostream>
#include "MATRIX.h"
#include <vector>
#include <string>
#include <random>
#include <iomanip>
#include <math.h>
#include <omp.h>
#include <fstream>

#ifndef NUG_H
#define NUG_H
class nugget {

private:
	
	std::vector<mat> weight; // vector of matrices for the weights of each layer
	std::vector<mat> bias; // vector of matrices for the biases of each layer
	std::vector<int> layer; // vector ints describing the layers
	//std::string activation; // Type of activation function for hidden layers
	//std::string out_activation; // typer of activation for output layer
	//std::string init; // type of weight initialization
	int Ninput, Noutput; // number of inputs and outputs neurons
	int activF; //Type of activation function used during training (set during training)
	mat xav_uni(int sizerow, int sizecol, int inputs, int outputs);
	mat xav_norm(int sizerow, int sizecol, int inputs, int outputs);
	mat ReLu(mat a);
	mat ReLuPr(mat a);
	mat sigmoid(mat a);
	mat sigmoidPr(mat a);
	friend double sig(double);
	mat OneHT(mat y);
	mat softmax(mat A);
	void accuracy(mat A, mat y);
	void save(std::string);
	void save();

public:
	// -- constructors --
	nugget(int inputs, int outputs, std::vector<int> hid_layers, std::string init);
	nugget(std::string read); // initialize using a save file of a previously trained NN
	// -- methods --

	void train(const mat& data, const mat& labels, int it, std::string activ, std::string o_activ, double alpha);
	void train(const mat& data, const mat& labels, int it, std::string activ, std::string o_activ, double alpha, std::string);
	void run(const mat& data, const mat& labels);
	void run(const mat& data);
	

};

nugget::nugget(std::string read) {
	std::vector<double> tempvec;
	std::vector<std::vector<double>> tempmat;
	std::ifstream savefile;
	savefile.open(read);
	int tempInt, temp2;
	double tempDouble;
	savefile >> tempInt;
	this->activF = tempInt;
	savefile >> tempInt;
	this->Ninput = tempInt;
	savefile >> tempInt;
	this->Noutput = tempInt;
	savefile >> temp2;
	//reading layer map
	for (int i = 0; i < temp2; i++) {
		savefile >> tempInt;
		this->layer.push_back(tempInt);
	}

	//Reading weights

	tempvec.resize(this->Ninput);
	tempmat.resize(this->layer[0]);
	for (int i = 0; i < this->layer[0]; i++) {
		for (int j = 0; j < this->Ninput; j++) {
			savefile >> tempvec[j];
		}
		tempmat[i] = tempvec;
	}
	this->weight.push_back(tempmat);
	tempvec.clear();
	tempmat.clear();
	
	
	for (int l = 1; l < layer.size(); l++) {
		tempvec.resize(this->layer[l-1]);
		tempmat.resize(this->layer[l]);
		for (int i = 0; i < this->layer[l]; i++) {
			for (int j = 0; j < this->layer[l - 1]; j++) {
				savefile >> tempvec[j];
			}
			tempmat[i] = tempvec;
		}
		this->weight.push_back(tempmat);
		tempvec.clear();
		tempmat.clear();
	}


	tempvec.resize(this->layer.back());
	tempmat.resize(this->Noutput);
	for (int i = 0; i < this->Noutput; i++) {
		for (int j = 0; j < this->layer.back(); j++) {
			savefile >> tempvec[j];
		}
		tempmat[i] = tempvec;
	}
	this->weight.push_back(tempmat);
	tempvec.clear();
	tempmat.clear();

	// reading biases

	tempvec.resize(this->layer[0]);
	for (int i = 0; i < this->layer[0]; i++) {
		savefile >> tempvec[i];
	}
	this->bias.push_back(tempvec);
	tempvec.clear();
	

	for (int l = 1; l < layer.size(); l++) {
		tempvec.resize(this->layer[l]);
		
		for (int i = 0; i < this->layer[l]; i++) {
			savefile >> tempvec[i];
			
		}
		this->bias.push_back(tempvec);
		tempvec.clear();
		
	}

	tempvec.resize(this->Noutput);
	for (int i = 0; i < this->Noutput; i++) {
		savefile >> tempvec[i];
	}
	this->bias.push_back(tempvec);
	tempvec.clear();

	savefile.close();
}

nugget::nugget(int inputs, int outputs, std::vector<int> hid_layers, std::string init) {// constructor to initialize the neural network with specified shape
	bool uni = true;
	if (init == "normal") {
		uni = false;
	}
	else if (init != "uniform") {
		std::cout << "invalid weight initialization constructor argument" << std::endl;
		exit(0);
	}
	/*std::cout << "mk1\n\n";*/
	this->layer = hid_layers;
	this->Ninput = inputs;
	this->Noutput = outputs;
	/*std::cout << "mk2\n\n";*/
	this->weight.resize(hid_layers.size() + 1);
	this->bias.resize(hid_layers.size() + 1);
	
	if (uni) {
		this->weight[0] = xav_uni(hid_layers[0], inputs, inputs, outputs);
	}
	else {
		this->weight[0] = xav_norm(hid_layers[0], inputs, inputs, outputs);
	}
	
	mat fstb("zeros", hid_layers[0], 1);
	/*std::cout << "mk3\n\n";*/
	
	this -> bias[0] = fstb;
	/*std::cout << "mk4\n\n";*/
	// creating weight and bias matrix for each hidden layer past layer one
	for (int i = 1; i < hid_layers.size(); i++) {
		if (uni) {
			this->weight[i] = xav_uni(hid_layers[i], hid_layers[i - 1], inputs, outputs);
		}
		else {
			this->weight[i] = xav_norm(hid_layers[i], hid_layers[i - 1], inputs, outputs);
		}
		mat btemp("zeros", hid_layers[i], 1);
		this->bias[i] = btemp;
		
	}
	// weights and bias to go from the last hidden layer to the output layer
	if (uni) {
		this->weight[hid_layers.size()] = xav_uni(outputs, hid_layers[hid_layers.size() - 1], inputs, outputs);
	}
	else {
		this->weight[hid_layers.size()] = xav_norm(outputs, hid_layers[hid_layers.size() - 1], inputs, outputs);
	}
	
	mat lastb("zeros", outputs, 1);
	
	this->bias[hid_layers.size()] = (lastb);
	
}

mat nugget::xav_uni(int sizerow, int sizecol, int inputs, int outputs) { //uniform xavier initialization

	double xavier = sqrt(6/((double)inputs + (double)outputs));
	mat W1("rand", -xavier, xavier, sizerow, sizecol);
	return W1;
}

mat nugget::xav_norm(int sizerow, int sizecol, int inputs, int outputs) { //normal xavier initialization

	double xavier = sqrt(2 / ((double)inputs + (double)outputs));
	mat W1("randN", 0, xavier, sizerow, sizecol);
	return W1;
}

mat nugget::OneHT(mat y) {
	mat onehot("zeros", 10, y.rows());
	for (int i = 0; i < y.rows(); i++) {
		onehot(y(i, 0), i) = 1;
	}

	return onehot;
}

mat nugget::ReLu(mat a) {

	mat temp = a;
#pragma omp parallel for 
	for (int i = 0; i < a.rows(); i++) {
		for (int j = 0; j < a.cols(); j++) {
			if (a(i, j) <= 0) {
				temp(i, j) = 0;
			}
			else {
				temp(i, j) = a(i, j);
			}
		}
	}

	return temp;
}

mat nugget::ReLuPr(mat a) {

	mat temp = a;
#pragma omp parallel for
	for (int i = 0; i < a.rows(); i++) {
		for (int j = 0; j < a.cols(); j++) {
			if (a(i, j) <= 0) {
				temp(i, j) = 0;
			}
			else {
				temp(i, j) = 1;
			}
		}
	}

	return temp;
}

double sig(double a) {
	double b;
	b = 1 / (1 + (exp(-a)));
	return b;
}

mat nugget::sigmoid(mat a) {

	mat temp = a;
#pragma omp parallel for
	for (int i = 0; i < a.rows(); i++) {
		for (int j = 0; j < a.cols(); j++) {
			
			temp(i, j) = sig(a(i, j));
			
		}
	}

	return temp;

}

mat nugget::sigmoidPr(mat a) {

	mat temp = a;
#pragma omp parallel for
	for (int i = 0; i < a.rows(); i++) {
		for (int j = 0; j < a.cols(); j++) {

			temp(i, j) = sig(a(i, j))*(1- sig(a(i, j)));

		}
	}

	return temp;

}

mat nugget::softmax(mat A) {

	
	mat B = A.xp() / A.xp().sum("cols");
	return B;
}

void nugget::accuracy(mat A, mat Y) {
	int m = A.cols();

	int sum = 0;
	double accuracy;
	double tempmax = 0;
	int indxmax1, indxmax2;
	//#pragma omp parallel for
	for (int i = 0; i < A.cols(); i++) {
		tempmax = 0;
		for (int j = 0; j < A.rows(); j++) {
			if (A(j, i) > tempmax) {
				tempmax = A(j, i);
				indxmax1 = j;
			}
			if (Y(j, i) == 1) {

				indxmax2 = j;
			}
		}
		if (indxmax1 == indxmax2) {
			sum++;

		}
	}
	accuracy = sum / (double)m;
	std::cout << "accuracy is " << accuracy << "\n\n";
}

void nugget::save() {
	char tab = 9;
	std::ofstream savefile;
	savefile.open("save.nn");
	savefile << this->activF << "\n";
	savefile << this->Ninput << "\n";
	savefile << this->Noutput << "\n";
	savefile << this->layer.size() << "\n";
	for (int i = 0; i < this->layer.size(); i++) {
		savefile << this->layer[i];
		if (i < this->layer.size() - 1) {
			savefile << tab;
		}

	}
	savefile << "\n";

	for (int i = 0; i < this->weight.size(); i++) { //Write weight matrices
		if (i != 0) {
			savefile << "\n";
		}
		for (int j = 0; j < this->weight[i].rows(); j++) {
			if (j != 0) {
				savefile << "\n";
			}
			for (int k = 0; k < this->weight[i].cols(); k++) {
				if (k != 0) {
					savefile << tab;
				}
				savefile << this->weight[i](j, k);
			}
		}
	}
	for (int i = 0; i < this->bias.size(); i++) { //Write weight matrices

		savefile << "\n";

		for (int j = 0; j < this->bias[i].rows(); j++) {
			if (j != 0) {
				savefile << tab;
			}
			savefile << this->bias[i](j, 0);

		}
	}
	savefile.close();
}

void nugget::save(std::string filename) {
	char tab = 9;
	std::ofstream savefile;
	savefile.open(filename);
	savefile << this->activF << "\n";
	savefile << this->Ninput << "\n";
	savefile << this->Noutput << "\n";
	savefile << this->layer.size() << "\n";
	for (int i = 0; i < this->layer.size(); i++) {
		savefile << this->layer[i];
		if (i < this->layer.size() - 1) {
			savefile << tab;
		}

	}
	savefile << "\n";

	for (int i = 0; i < this->weight.size(); i++) { //Write weight matrices
		if (i != 0) {
			savefile << "\n";
		}
		for (int j = 0; j < this->weight[i].rows(); j++) {
			if (j != 0) {
				savefile << "\n";
			}
			for (int k = 0; k < this->weight[i].cols(); k++) {
				if (k != 0) {
					savefile << tab;
				}
				savefile << this->weight[i](j, k);
			}
		}
	}
	for (int i = 0; i < this->bias.size(); i++) { //Write weight matrices

		savefile << "\n";

		for (int j = 0; j < this->bias[i].rows(); j++) {
			if (j != 0) {
				savefile << tab;
			}
			savefile << this->bias[i](j, 0);

		}
	}
	savefile.close();
}


void nugget::train(const mat& Data, const mat& labels, int it, std::string activ, std::string o_activ, double alpha) {

	int m; // size of data sample
	int activation;

	// seting condition for selected activation function
	if (activ == "ReLu") {
		activation = 0;
		this->activF = 0;
	}
	else if (activ == "sigmoid") {
		activation = 1;
		this->activF = 1;
	}
	else {
		std::cout << "invalid activation function argument for training function, please select either 'ReLu' or 'sigmoid'." << std::endl;
		exit(0);
	}
	// normalize and process input data
	
	mat data = Data;
	if (data.rows() == this->Ninput) {
		m = data.cols();
		data = data / data.max();
	}
	else if (data.cols() == this->Ninput) {
		data = data.T();
		m = data.cols();
		data = data / data.max();
	}
	else {
		std::cout << "Dimensions of the data matrix are "<<data.rows()<<" by "<<data.cols()<<", at least one of those dimensions should match the specificed input layer size for this nugget which is"<< this->Ninput <<"." << std::endl;
		exit(0);
	}


	// 1-hot labels
	
	mat lab = OneHT(labels);

	std::vector<mat> A, Z, dW, dZ, db;
	A.resize(layer.size() + 1);
	Z.resize(layer.size() + 1);
	dW.resize(layer.size() + 1);
	dZ.resize(layer.size() + 1);
	db.resize(layer.size() + 1);
	int epoch = 0;
	
	while (epoch < it) {
		// forward propagation
		Z[0] = this->weight[0] * data + this->bias[0];
		if (activation) {
			
			A[0] = sigmoid(Z[0]);
		}
		else {

			A[0] = ReLu(Z[0]);

		}
		
		
		for (int i = 1; i < this->layer.size(); i++) {
			Z[i] = this->weight[i] * A[i - 1] + this->bias[i];

			if (activation) {

				A[i] = sigmoid(Z[i]);
			}
			else {

				A[i] = ReLu(Z[i]);

			}

		}
		/*std::cout << "W3 is "<< this->weight[layer.size()].rows() << " by " << this->weight[layer.size()].cols() <<"\n\n";
		std::cout << "A2 is " << A[layer.size() - 1].rows() << " by " << A[layer.size() - 1].cols() << "\n\n";
		std::cout << "b3 is " << this->bias[layer.size()].rows() << " by " << this->bias[layer.size()].cols() << "\n\n";*/

		Z[layer.size()] = this->weight[layer.size()] * A[layer.size() - 1] + this->bias[layer.size()];
		
		A[layer.size()] = softmax(Z[layer.size()]); // output layer

		// back propagation
		
		dZ[layer.size()] = A[layer.size()] - lab;
		dW[layer.size()] = 1 / (double)m * dZ[layer.size()] * A[layer.size() - 1].T();
		db[layer.size()] = 1 / (double)m * dZ[layer.size()].sum("rows");
		
		for (int i = layer.size() - 1; i > 0; i--) {

			if (activation) {

				dZ[i] = this->weight[i + 1].T() * dZ[i + 1] ^ sigmoidPr(Z[i]);
			}
			else {

				dZ[i] = this->weight[i + 1].T() * dZ[i + 1] ^ ReLuPr(Z[i]);

			}

			
			//std::cout << "mk3\n\n";
			dW[i] = 1 / (double)m * dZ[i] * A[i-1].T();
			//std::cout << "mk4\n\n";
			db[i] = 1 / (double)m * dZ[i].sum("rows");
		}

		if (activation) {

			dZ[0] = this->weight[1].T() * dZ[1] ^ sigmoidPr(Z[0]);
		}
		else {

			dZ[0] = this->weight[1].T() * dZ[1] ^ ReLuPr(Z[0]);

		}

		
		//std::cout << "mk3\n\n";
		dW[0] = 1 / (double)m * dZ[0] * data.T();
		//std::cout << "mk4\n\n";
		db[0] = 1 / (double)m * dZ[0].sum("rows");
		
		for (int i = 0; i < this->layer.size() + 1; i++) {
			
			this->weight[i] = weight[i] - alpha * dW[i];
			
			this->bias[i] = bias[i] - alpha * db[i];
		}
		
		if (epoch % 10 == 0) {
			std::cout << "Epoch: " << epoch << "\n";
			accuracy(A[layer.size()], lab);
			

		}

		epoch++;
	}

	std::cout << "Saving model. " << epoch << "\n";
	save();

}

void nugget::train(const mat& Data, const mat& labels, int it, std::string activ, std::string o_activ, double alpha, std::string filename) {

	int m; // size of data sample
	int activation;

	// seting condition for selected activation function
	if (activ == "ReLu") {
		activation = 0;
		this->activF = 0;
	}
	else if (activ == "sigmoid") {
		activation = 1;
		this->activF = 1;
	}
	else {
		std::cout << "invalid activation function argument for training function, please select either 'ReLu' or 'sigmoid'." << std::endl;
		exit(0);
	}
	// normalize and process input data

	mat data = Data;
	if (data.rows() == this->Ninput) {
		m = data.cols();
		data = data / data.max();
	}
	else if (data.cols() == this->Ninput) {
		data = data.T();
		m = data.cols();
		data = data / data.max();
	}
	else {
		std::cout << "Dimensions of the data matrix are " << data.rows() << " by " << data.cols() << ", at least one of those dimensions should match the specificed input layer size for this nugget which is" << this->Ninput << "." << std::endl;
		exit(0);
	}


	// 1-hot labels

	mat lab = OneHT(labels);

	std::vector<mat> A, Z, dW, dZ, db;
	A.resize(layer.size() + 1);
	Z.resize(layer.size() + 1);
	dW.resize(layer.size() + 1);
	dZ.resize(layer.size() + 1);
	db.resize(layer.size() + 1);
	int epoch = 0;

	while (epoch < it) {
		// forward propagation
		Z[0] = this->weight[0] * data + this->bias[0];
		if (activation) {

			A[0] = sigmoid(Z[0]);
		}
		else {

			A[0] = ReLu(Z[0]);

		}


		for (int i = 1; i < this->layer.size(); i++) {
			Z[i] = this->weight[i] * A[i - 1] + this->bias[i];

			if (activation) {

				A[i] = sigmoid(Z[i]);
			}
			else {

				A[i] = ReLu(Z[i]);

			}

		}
		/*std::cout << "W3 is "<< this->weight[layer.size()].rows() << " by " << this->weight[layer.size()].cols() <<"\n\n";
		std::cout << "A2 is " << A[layer.size() - 1].rows() << " by " << A[layer.size() - 1].cols() << "\n\n";
		std::cout << "b3 is " << this->bias[layer.size()].rows() << " by " << this->bias[layer.size()].cols() << "\n\n";*/

		Z[layer.size()] = this->weight[layer.size()] * A[layer.size() - 1] + this->bias[layer.size()];

		A[layer.size()] = softmax(Z[layer.size()]); // output layer

		// back propagation

		dZ[layer.size()] = A[layer.size()] - lab;
		dW[layer.size()] = 1 / (double)m * dZ[layer.size()] * A[layer.size() - 1].T();
		db[layer.size()] = 1 / (double)m * dZ[layer.size()].sum("rows");

		for (int i = layer.size() - 1; i > 0; i--) {

			if (activation) {

				dZ[i] = this->weight[i + 1].T() * dZ[i + 1] ^ sigmoidPr(Z[i]);
			}
			else {

				dZ[i] = this->weight[i + 1].T() * dZ[i + 1] ^ ReLuPr(Z[i]);

			}


			//std::cout << "mk3\n\n";
			dW[i] = 1 / (double)m * dZ[i] * A[i - 1].T();
			//std::cout << "mk4\n\n";
			db[i] = 1 / (double)m * dZ[i].sum("rows");
		}

		if (activation) {

			dZ[0] = this->weight[1].T() * dZ[1] ^ sigmoidPr(Z[0]);
		}
		else {

			dZ[0] = this->weight[1].T() * dZ[1] ^ ReLuPr(Z[0]);

		}


		//std::cout << "mk3\n\n";
		dW[0] = 1 / (double)m * dZ[0] * data.T();
		//std::cout << "mk4\n\n";
		db[0] = 1 / (double)m * dZ[0].sum("rows");

		for (int i = 0; i < this->layer.size() + 1; i++) {

			this->weight[i] = weight[i] - alpha * dW[i];

			this->bias[i] = bias[i] - alpha * db[i];
		}

		if (epoch % 10 == 0) {
			std::cout << "Epoch: " << epoch << "\n";
			accuracy(A[layer.size()], lab);


		}

		epoch++;
	}

	std::cout << "Saving model. "<< "\n";
	save(filename);

}

void nugget::run(const mat& Data, const mat& labels) {
	// runing the model is just the forward propagation step with a new dataset. This function takes labels and performs an accuracy calculation at the end.

	int m; // size of data sample
	
	// normalize and process input data

	mat data = Data;
	if (data.rows() == this->Ninput) {
		m = data.cols();
		data = data / data.max();
	}
	else if (data.cols() == this->Ninput) {
		data = data.T();
		m = data.cols();
		data = data / data.max();
	}
	else {
		std::cout << "Dimensions of the data matrix are " << data.rows() << " by " << data.cols() << ", at least one of those dimensions should match the specificed input layer size for this nugget which is" << this->Ninput << "." << std::endl;
		exit(0);
	}


	// 1-hot labels

	mat lab = OneHT(labels);

	std::vector<mat> A, Z;
	A.resize(layer.size() + 1);
	Z.resize(layer.size() + 1);
	
	

	// forward propagation
	Z[0] = this->weight[0] * data + this->bias[0];
	if (this->activF) {

		A[0] = sigmoid(Z[0]);
	}
	else {

		A[0] = ReLu(Z[0]);

	}


	for (int i = 1; i < this->layer.size(); i++) {
		Z[i] = this->weight[i] * A[i - 1] + this->bias[i];

		if (this->activF) {

			A[i] = sigmoid(Z[i]);
		}
		else {

			A[i] = ReLu(Z[i]);

		}

	}
	

	Z[layer.size()] = this->weight[layer.size()] * A[layer.size() - 1] + this->bias[layer.size()];

	A[layer.size()] = softmax(Z[layer.size()]); // output layer

	accuracy(A[layer.size()], lab);

}

void nugget::run(const mat& Data) {
	// runing the model is just the forward propagation step with a new dataset. This function takes labels and performs an accuracy calculation at the end.

	int m; // size of data sample

	mat data = Data;
	if (data.rows() == this->Ninput) {
		m = data.cols();
		data = data / data.max();
	}
	else if (data.cols() == this->Ninput) {
		data = data.T();
		m = data.cols();
		data = data / data.max();
	}
	else {
		std::cout << "Dimensions of the data matrix are " << data.rows() << " by " << data.cols() << ", at least one of those dimensions should match the specificed input layer size for this nugget which is" << this->Ninput << "." << std::endl;
		exit(0);
	}


	std::vector<mat> A, Z;
	A.resize(layer.size() + 1);
	Z.resize(layer.size() + 1);
	

	
	// forward propagation
	Z[0] = this->weight[0] * data + this->bias[0];
	if (this->activF) {

		A[0] = sigmoid(Z[0]);
	}
	else {

		A[0] = ReLu(Z[0]);

	}


	for (int i = 1; i < this->layer.size(); i++) {
		Z[i] = this->weight[i] * A[i - 1] + this->bias[i];

		if (this->activF) {

			A[i] = sigmoid(Z[i]);
		}
		else {

			A[i] = ReLu(Z[i]);

		}

	}


	Z[layer.size()] = this->weight[layer.size()] * A[layer.size() - 1] + this->bias[layer.size()];

	A[layer.size()] = softmax(Z[layer.size()]); // output layer

	

}

#endif