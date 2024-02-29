#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <iomanip>
#include <math.h>
#include <omp.h>

#ifndef MAT_H
#define MAT_H

class mat {

	private:
		unsigned int n, m;
		std::vector<std::vector<double>> matrix;
	public:
		// -- constructors --
		mat(std::string, unsigned int);
		mat(std::string , unsigned int , unsigned int );
		mat(std::string , double , double , unsigned int , unsigned int );
		mat(double , unsigned int , unsigned int );
		mat(std::vector<std::vector<double>>& );
		mat(std::vector<std::vector<unsigned int>>&);
		mat(std::vector<std::vector<int>>&);
		mat(std::vector<std::vector<float>>&);
		mat(std::vector<double>&);
		mat(std::vector<unsigned int>&);
		mat(std::vector<int>&);
		mat(std::vector<float>&);
		mat();
		// -- operators --
		// matrix operations
		mat operator+(const mat& ); // addition
		mat operator-(const mat& ); // substraction
		mat operator*(const mat& ); // multiplication
		mat operator^(const mat&); // broadcasting/Hadamard product
		// matrix index
		double& operator()(const unsigned int& , const unsigned int& );
		// matrix comparison
		bool operator==(const mat& );
		// matrix assignment
		void operator=(const mat& );
		// scalar operations
		/*mat operator+(const double );
		mat operator-(const double );*/
		friend mat operator*(const double a, const mat& A);
		friend mat operator*(const mat& A, const double a); 
		mat operator/(const double );
		mat operator/(mat A); // dividing matrix by 1D vector by row or column 
		// -- utilities -- 
		mat T(); // transpose
		void print(); // print matrix
		unsigned int rows(); // returns number of rows
		unsigned int cols(); // returns number of column
		mat sum(std::string ); // sum along axis
		double sum(); // sum all cells
		double max(); // returns the largest element of the matrix
		double min(); // returns the smallest element of the matrix
		mat xp(); // returns e^x of every matrix elements
};

mat::mat(std::string arg, unsigned int M) {
	if (arg == "I") {
		m = M;
		n = M;
		std::vector<std::vector<double>> vec(M, std::vector<double>(M, 0));
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (i == j) {
					vec[i][j] = 1;
				}

			}
		}
		matrix = vec;
	}
	else {
		std::cout << "invalid matrix constructor argument" << std::endl;
		exit(0);
	}
	
}


mat::mat(std::string arg, unsigned int M, unsigned int N) {
	// read the argument
	
	if (arg == "zeros") { // null matrix
		std::vector<std::vector<double>> vec(M, std::vector<double>(N, 0));
		matrix = vec;
		m = M;
		n = N;
	}
	else if (arg == "I") { // identity matrix
		if (N != M) {
			std::cout << "Cannot generate identity rectangular matrix, m must be equal to n." << std::endl;
			exit(0);
		}
		m = M;
		n = M;
		std::vector<std::vector<double>> vec(M, std::vector<double>(M, 0));
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (i == j) {
					vec[i][j] = 1;
				}
				
			}
		}
		
		matrix = vec;
	}
	else { // error
		std::cout << "invalid matrix constructor argument" << std::endl;
		exit(0);
	}


}

mat::mat(std::string arg, double a, double b, unsigned int M, unsigned int N) {
	// read the argument

	if (arg == "rand") { // intialize matrix with random numbers
		std::vector<std::vector<double>> vec(M, std::vector<double>(N, 0));

		m = M;
		n = N;

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(a, b);

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				vec[i][j] = dis(gen);
			}
		}

		matrix = vec;
	}
	else if (arg == "randN") { // intialize matrix with random numbers
		std::vector<std::vector<double>> vec(M, std::vector<double>(N, 0));
		// for normal distributions, a is the mean value, and b is the standard deviation
		m = M;
		n = N;

		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> dis(a, b);

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				vec[i][j] = dis(gen);
			}
		}

		matrix = vec;
	}

	else { // error
		std::cout << "invalid matrix constructor arguments" << std::endl;
		exit(0);
	}


}

mat::mat(double a, unsigned int M, unsigned int N) { // constructor to initialize all values as 'a' for given matrix size
	
	std::vector<std::vector<double>> vec(M, std::vector<double>(N, a));

	m = M;
	n = N;

	matrix = vec;

}

mat::mat(std::vector<std::vector<double>>& M) { // contructor which accepts std::vectors
	m = M.size();
	n = M[0].size();
	matrix = M;
}

mat::mat(std::vector<std::vector<unsigned int>>& M) { // contructor which accepts std::vector<unsigned int>
	m = M.size();
	n = M[0].size();
	matrix.resize(m, std::vector<double>(n));
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			matrix[i][j] = double(M[i][j]);
		}
	}
	
}

mat::mat(std::vector<std::vector<int>>& M) { // contructor which accepts std::vector<int>
	m = M.size();
	n = M[0].size();
	matrix.resize(m, std::vector<double>(n));
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			matrix[i][j] = double(M[i][j]);
		}
	}

}

mat::mat(std::vector<std::vector<float>>& M) { // contructor which accepts std::vector<float>
	m = M.size();
	n = M[0].size();
	matrix.resize(m, std::vector<double>(n));
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			matrix[i][j] = double(M[i][j]);
		}
	}

}

mat::mat(std::vector<float>& M) { // contructor which accepts 1D std::vector<float>
	m = M.size();
	n = 1;
	matrix.resize(m, std::vector<double>(n));
	for (int i = 0; i < m; i++) {
		
		matrix[i][0] = double(M[i]);
		
	}

}

mat::mat(std::vector<int>& M) { // contructor which accepts 1D std::vector<int>
	m = M.size();
	n = 1;
	matrix.resize(m, std::vector<double>(n));
	for (int i = 0; i < m; i++) {

		matrix[i][0] = double(M[i]);

	}

}

mat::mat(std::vector<unsigned int>& M) { // contructor which accepts 1D std::vector<unsigned int>
	m = M.size();
	n = 1;
	matrix.resize(m, std::vector<double>(n));
	for (int i = 0; i < m; i++) {

		matrix[i][0] = double(M[i]);

	}

}

mat::mat(std::vector<double>& M) { // contructor which accepts 1D std::vector<double>
	m = M.size();
	n = 1;
	matrix.resize(m, std::vector<double>(n));
	for (int i = 0; i < m; i++) {

		matrix[i][0] = double(M[i]);

	}

}

mat::mat() { //default constructor

	std::vector<std::vector<double>> vec;
	matrix = vec;
	m = 0;
	n = 0;

}

void mat::print() { // matrix print function
	std::cout << std::setprecision(3);
	unsigned int N;
	if (n > 20) {
		std::cout << "the matrix it too wide to print, printing only the first 20 columns: \n";
		N = 20;
	}
	else {
		N = n;
	}

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < N; j++) {
			if (j == 0) {
				std::cout << "\n| " << std::setw(8) << matrix[i][j] << (j == N - 1 ? " |" : "");
			}
			else {
				std::cout << std::setw(8)<< matrix[i][j] << (j == N - 1 ? " |" : "");
			}
			
		}
	}
	std::cout << '\n';
}

mat mat::T() { // matrix transpose function
	mat transpose("zeros", this->n, this->m); // needs destructor
	#pragma omp parallel for
	for (int i = 0; i < this->m; i++) {
		for (int j = 0; j < this->n; j++) {
			transpose(j, i) = this->matrix[i][j];
		}
	}


	return transpose;
}

mat mat::operator+(const mat& B) { // matrix addition

	mat sum("zeros", this->m, this->n);

	if (this->m != B.m && this->n != B.n) {
		std::cout << "matrix addition error, matrix dimension mismatch.\n";
		exit(0);
	}
	else if (this->m == B.m && this->n == B.n) { // classic matrix addition
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				sum.matrix[i][j] = this->matrix[i][j] + B.matrix[i][j];
			}
		}
	}
	else if (this->m == B.m && B.n == 1) { // extrude the vetor B to match columns
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				sum.matrix[i][j] = this->matrix[i][j] + B.matrix[i][0];
			}
		}
	}
	else if (this->n == B.n && B.m == 1) {// extrude the vetor B to match rows
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				sum.matrix[i][j] = this->matrix[i][j] + B.matrix[0][j];
			}
		}
	}
	else {
		std::cout << "matrix addition error, matrix dimension mismatch.\n";
		exit(0);
	}

	return sum;
}

//mat mat::operator+(const double a) { // scalar addition
//	
//	mat sum("zeros", this->m, this->n);
//	for (int i = 0; i < m; i++) {
//		for (int j = 0; j < n; j++) {
//			sum.matrix[i][j] = this->matrix[i][j] + a;
//		}
//	}
//
//	return sum;
//}

mat mat::operator-(const mat& B) { // matrix substraction
	mat sum("zeros", this->m, this->n);
	
	if (this->m != B.m && this->n != B.n) {
		std::cout << "matrix substraction error, matrix dimension mismatch.\n";
		exit(0);
	}
	else if (this->m == B.m && this->n == B.n) { // classic matrix substraction
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				sum.matrix[i][j] = this->matrix[i][j] - B.matrix[i][j];
			}
		}
	}
	else if (this->m == B.m && B.n == 1) { // extrude the vetor B to match columns (Broadcasting)
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				sum.matrix[i][j] = this->matrix[i][j] - B.matrix[i][0];
			}
		}
	}
	else if (this->n == B.n && B.m == 1) {// extrude the vetor B to match rows (Broadcasting)
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				sum.matrix[i][j] = this->matrix[i][j] - B.matrix[0][j];
			}
		}
	}
	else {
		std::cout << "matrix substraction error, matrix dimension mismatch.\n";
		exit(0);
	}

	return sum;
}

//mat mat::operator-(const double a) { // scalar substraction
//	
//	mat sub("zeros", this->m, this->n);
//	for (int i = 0; i < m; i++) {
//		for (int j = 0; j < n; j++) {
//			sub.matrix[i][j] = this->matrix[i][j] - a;
//		}
//	}
//
//	return sub;
//}

void mat::operator=(const mat& B) { // overload of assignment operator
	this->m = B.m;
	this->n = B.n;
	this->matrix = B.matrix;
}

bool mat::operator==(const mat& B) { // overlaod of comparison operator
	if (this->matrix == B.matrix) { return true; }
	else { return false; }
}

mat mat::operator*(const mat& B) { // matrix multiplication
	if (this->n != B.m) {
		std::cout << "matrix multiplication error, matrix dimension mismatch.\n";
		exit(0);
	}
	double sum;
	mat mult("zeros", this->m, B.n);
	#pragma omp parallel for private (sum)
	for (int i = 0; i < this->m; i++) {
		for (int j = 0; j < B.n; j++) {
			sum = 0;
			for (int k = 0; k < this->n; k++) {
				sum += this->matrix[i][k] * B.matrix[k][j];
			}
			mult.matrix[i][j] = sum;
		}
	}
	mult.n = mult.matrix[0].size();
	mult.m = mult.matrix.size();
	return mult;
}

mat mat::operator^(const mat& B) { // broadcasting and Hadamard product

	mat brod("zeros", this->m, this->n);

	if (this->m == B.m && this->n == B.n) { // Hadamard product
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				brod.matrix[i][j] = this->matrix[i][j] * B.matrix[i][j];
			}
		}
	}
	else if (this->m == B.m && B.n == 1) {
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				brod.matrix[i][j] = this->matrix[i][j] * B.matrix[i][0];
			}
		}
	}
	else if (this->n == B.n && B.m == 1) {
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				brod.matrix[i][j] = this->matrix[i][j] * B.matrix[0][j];
			}
		}
	}
	else {
		std::cout << "matrix broadcasting error, matrix dimension mismatch.\n";
		exit(0);
	}

	return brod;
}

mat operator*(const mat& A, const double a) { // scalar multiplication

	mat sub("zeros", A.m, A.n);
	#pragma omp parallel for
	for (int i = 0; i < A.m; i++) {
		for (int j = 0; j < A.n; j++) {
			sub.matrix[i][j] = A.matrix[i][j] * a;
		}
	}

	return sub;
}

mat operator*(const double a, const mat& A) { // scalar multiplication

	mat sub("zeros", A.m, A.n);
	#pragma omp parallel for
	for (int i = 0; i < A.m; i++) {
		for (int j = 0; j < A.n; j++) {
			sub.matrix[i][j] = A.matrix[i][j] * a;
		}
	}

	return sub;
}

mat mat::operator/(const double a) { // scalar division

	mat sub("zeros", this->m, this->n);
	#pragma omp parallel for
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			sub.matrix[i][j] = this->matrix[i][j] / a;
		}
	}

	return sub;
}

mat mat::operator/(mat A) { // vector division- broadcasting

	mat sub("zeros", this->m, this->n);
	
	if (A.rows() == m && A.cols() == 1) { // divide each element of the matrix rows by corresponding row element of vector A
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				sub.matrix[i][j] = this->matrix[i][j] / A(i,0);
			}
		}
	}
	else if (A.cols() == n && A.rows() == 1) { // divide each element of the matrix columns by corresponding column element of vector A
		
		
		#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				
				sub.matrix[i][j] = this->matrix[i][j] / A(0, j);
			}
		}
	}
	else if (A.rows() == m && A.cols() == n) { // divide each element of the matrix rows by corresponding row element of vector A
	#pragma omp parallel for
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				sub.matrix[i][j] = this->matrix[i][j] / A(i, j);
			}
		}
	}
	else if(A.rows() > 1 && A.cols() > 1 ){
		std::cout << "The divisor vector must have one of its dimensions be 1, it is currently a 2D matrix.\n";
		exit(0);
	}
	else {
		std::cout << "Division of matrix by vector error: mismatched dimensions.\n";
		exit(0);
	}
	

	return sub;
}

unsigned int mat::rows() {
	return this->m;
}

unsigned int mat::cols() {
	return this->n;
}

double& mat::operator()(const unsigned int& i, const unsigned int& j) {
	return this->matrix[i][j];
}

mat mat::sum(std::string arg) { // summing columns or rows 
	if (arg == "rows") {
		mat SUM("zeros", this->m, 1);
		double tempsum=0;
		#pragma omp parallel for reduction (+:tempsum)
		for (int i = 0; i < this->m; i++) {
			tempsum = 0;
			for (int j = 0; j < this->n; j++) {
				tempsum += this->matrix[i][j];
			}
			SUM(i, 0) = tempsum;
		}
		return SUM;
	}
	else if (arg == "cols") {
		mat SUM("zeros", 1, this->n);
		double tempsum = 0;
		#pragma omp parallel for reduction (+:tempsum)
		for (int i = 0; i < this->n; i++) {
			tempsum = 0;
			for (int j = 0; j < this->m; j++) {
				tempsum += this->matrix[j][i];
			}
			SUM(0, i) = tempsum;
		}
		return SUM;
	}
	else {
		std::cout << "improper argument for 'sum()' please use 'cols' to sum columns, or 'rows' to sum rows.\n";
		exit(0);
	}

	
}

double mat::sum() { // summing all matrix elements into a scalar value
	double sum = 0;
	#pragma omp parallel for reduction (+:sum)
	for (int i = 0; i < this->m; i++) {
		for (int j = 0; j < this->n; j++) {
			sum += this->matrix[i][j];
		}
	}
	return sum;
}

double mat::max() {
	if (this->m == 0 && this->n == 0) { return 0; }
	double max = this->matrix[0][0];
	for (int i = 0; i < this->m; i++) {
		for (int j = 0; j < this->n; j++) {
			if (max < this->matrix[i][j]) {
				max = this->matrix[i][j];
			}
		}
	}
	return max;

}

double mat::min() {
	if (this->m == 0 && this->n == 0) { return 0; }
	double min = matrix[0][0];
	for (int i = 0; i < this->m; i++) {
		for (int j = 0; j < this->n; j++) {
			if (min > matrix[i][j]) {
				min = matrix[i][j];
			}
		}
	}
	return min;

}

mat mat::xp() {
	mat temp("zeros", m, n);
	#pragma omp parallel for
	for (int i = 0; i < this->m; i++) {
		for (int j = 0; j < this->n; j++) {
			temp(i, j) = exp(this->matrix[i][j]);
		}
	}
	return temp;
}

#endif