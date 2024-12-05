#include <stdio.h>
#include <fstream>
#include <iostream>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;


const int DATA_SET_SIZE = 100000;
const int MOVES = 3;
const int PERMUTATIONS = MOVES * MOVES;
int NAB[PERMUTATIONS];
int dataSet[DATA_SET_SIZE];
const int threadsPerBlock = 256;
const int blocksPerGrid = (DATA_SET_SIZE + threadsPerBlock - 1) / threadsPerBlock;


// Generate a random set of integers each being in the range 0 to MOVES - 1
// Save numbers to file to ensure tests can be repeated.
void GenerateData() {
	ofstream out("data.dat", ios::out | ios::binary);
	for (int n = 0; n < DATA_SET_SIZE; n++) {
		int i = rand() % MOVES;
		out.write((char*)&i, sizeof(i));
	}
	out.close();
}

// I've found the initialise function isn't used in the rest of the code.

//void InitialiseNAB() { 
//	for (int n = 0; n < PERMUTATIONS; n++) {
//		NAB[n] = 0;
//	}
//} 


// Populate data array with contents of file
void GetData() {
	ifstream in("data.dat", ios::in | ios::binary);
	for (int n = 0; n < DATA_SET_SIZE; n++) {
		in.read((char*)&dataSet[n], sizeof(int));
	}
	in.close();
}


__device__ int GetIndex(int firstMove, int secondMove) { //device allows function to be called by global functions
		//if (firstMove == 0 && secondMove == 0) return 0;
		//if (firstMove == 0 && secondMove == 1) return 1;
		//if (firstMove == 0 && secondMove == 2) return 2;
		//if (firstMove == 1 && secondMove == 0) return 3;
		//if (firstMove == 1 && secondMove == 1) return 4;
		//if (firstMove == 1 && secondMove == 2) return 5;
		//if (firstMove == 2 && secondMove == 0) return 6;
		//if (firstMove == 2 && secondMove == 1) return 7;
		//if (firstMove == 2 && secondMove == 2) return 8;
	/*if (firstMove == 0) return firstMove + secondMove;
	if (firstMove == 1) return firstMove * MOVES + secondMove;
	if (firstMove == 2) return firstMove * MOVES + secondMove;*/ //atempt to shorten the calcluation
	return firstMove * MOVES + secondMove; //simplified into 1 calculation to make the comparison run faster. tested the calculation and it has the same return based on the int values.
	
}

void DisplayNAB() {
	int check = 0;
	cout << endl;
	for (int n = 0; n < PERMUTATIONS; n++) {
		cout << "Index " << n << " : " << NAB[n] << endl;
		check += NAB[n];
	}
	// Total should be one less than DATA_SET_SIZE as first value doesn't have a previous value to compare.
	cout << "Total : " << check << endl;
}




__global__ void PopulateNAB(int* dataSet, int* NAB, float* totals) {
	__shared__ int cache[threadsPerBlock];

	float data = 0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x; //input array offset
	int cacheIndex = threadIdx.x;
	cache[cacheIndex] = data;
	__syncthreads(); //sync function

	int i = blockDim.x / 2; //reduction kernal function
	while (i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0) {
		totals[blockIdx.x] - cache[0];
	}

	int index;
	int previous = dataSet[0];
	while(tid < DATA_SET_SIZE) { // main function
		index = GetIndex(previous, dataSet[tid]);
		atomicAdd(&NAB[index], 1); //addition of point values
		previous = dataSet[tid];
		data += dataSet[tid] * NAB[tid];
		tid += blockDim.x * gridDim.x;
	}

}

int main() {
	cudaEvent_t start, stop; //creating the timer.
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);



	srand(time(NULL));
	GenerateData();
	GetData();
	int* dev_dataSet; //initialise pointers
	int* dev_NAB;
	float* dev_totals;

	cudaMalloc((void**)&dev_dataSet, DATA_SET_SIZE * sizeof(int)); //allocate memory
	cudaMalloc((void**)&dev_NAB, PERMUTATIONS * sizeof(int));
	cudaMalloc((void**)&dev_totals, sizeof(float) * blocksPerGrid);
	cudaMemcpy(dev_dataSet, dataSet, DATA_SET_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_NAB, NAB, PERMUTATIONS * sizeof(int), cudaMemcpyHostToDevice);

	const int size = 100; // executes on 100 block. uses muliple processors on the gpu

	PopulateNAB << <blocksPerGrid, threadsPerBlock >> > (dev_dataSet, dev_NAB, dev_totals); //kernel function
	cudaMemcpy(NAB, dev_NAB, PERMUTATIONS * sizeof(int), cudaMemcpyDeviceToHost); //copies data
	cudaFree(dev_dataSet); //free up the memory.
	cudaFree(dev_NAB);
	cudaFree(dev_totals);
	//PopulateNAB();
	DisplayNAB();
	
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop); //stops the timer
	float elapseTime;
	cudaEventElapsedTime(&elapseTime, start, stop); //records the time between start and stop
	printf("Time to generate: %3.1f ms \n", elapseTime); 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
