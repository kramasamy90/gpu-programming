#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

#define BUFFER 1024

__global__ void mult1(int* A, int* B, int* E1, int p, int q, int r){ // E1 = AB
    __shared__ int col_B[BUFFER]; // j-th column of B.
    __shared__ int dot_prod[BUFFER];
    __shared__ int col_E1[BUFFER]; // j-th column of E.

    memset(col_E1, 0, BUFFER * sizeof(int));

    int j = blockIdx.x; // Column number in B and in E.
    int k = threadIdx.x; // Row number in B and column number in A.

    col_B[k] = B[k * r + j];
    __syncthreads();

    for(int i = 0; i < p; i++){ // i is row number in A and E.
        col_E1[i] = 0;
        __syncthreads();
        dot_prod[k] = A[i * q + k] * col_B[k];
        __syncthreads();
        atomicAdd(&col_E1[i], dot_prod[k]);
        __syncthreads();
        E1[i * r + j] = col_E1[i];
        __syncthreads();
    }
}

__global__ void mult2(int* C, int* D, int* E2, int p, int q, int r){ // E2 = CD.T
    __shared__ int row_D[BUFFER]; // j-th row of D.
    __shared__ int dot_prod[BUFFER];
    __shared__ int col_E2[BUFFER]; // j-th column of E.

    memset(col_E2, 0, BUFFER * sizeof(int));

    int j = blockIdx.x; // Row number in D and in E.
    int k = threadIdx.x; // Column number in D and column number in C.

    row_D[k] = D[j *q + k]; // j-th row of D = j-th column of D.T.
    __syncthreads();

    for(int i = 0; i < p; i++){ // i is row number in A and E.
        col_E2[i] = 0;
        __syncthreads();
        dot_prod[k] = C[i * q + k] * row_D[k];
        __syncthreads();
        atomicAdd(&col_E2[i], dot_prod[k]);
        __syncthreads();
        E2[i * r + j] = col_E2[i];
        __syncthreads();
    }
}

__global__ void add(int* E1, int* E2, int* E, int p, int r){
    int i = blockIdx.x;
    int j = threadIdx.x;
    E[i * r + j] = E1[i * r + j] + E2[i * r +j];
}

// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
    int* d_matrixE1;
	cudaMalloc(&d_matrixE1, p * r * sizeof(int));
    mult1<<<r, q>>>(d_matrixA, d_matrixB, d_matrixE1, p, q, r);

    int* d_matrixE2;
	cudaMalloc(&d_matrixE2, p * r * sizeof(int));
    mult2<<<r, q>>>(d_matrixC, d_matrixD, d_matrixE2, p, q, r);

    add<<<p, r>>>(d_matrixE1, d_matrixE2, d_matrixE, p, r);
	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	
