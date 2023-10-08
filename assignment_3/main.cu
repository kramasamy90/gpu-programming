/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/




__global__ void d_level_0(int V, int *d_apr, bool* d_is_active, int* right){
    // Get the range of nodes in zeroeth level.
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id >= V) return;
    if(d_apr[id] == 0) {
        d_is_active[id] = true;
        atomicMax(right, id);
    }
} 

    
__global__ void d_get_aid(int V, int* d_offset, int* d_csrList, int* d_aid, int* d_apr, bool* d_is_active, int* d_activeVertex, int* left, int* new_left, int* right, int* new_right, int level){
    int id = blockDim.x * blockIdx.x + threadIdx.x + *left;
    int num_active_threads = *right - *left + 1;
    if(id <= *right){
        if(d_is_active[id]){
            // Increment no. of active vertex at current level.
            atomicAdd(&d_activeVertex[level], 1);

            // Get active-indegree for next level.
            // Get max destination: new_right.
            // Get min destination: new_left
            int max_destination = 0;
            int min_destination = V+10;
            *new_left = V + 10;
            for(int i = d_offset[id]; i < d_offset[id+1]; i++){
                int destination = d_csrList[i];
                min_destination = min(min_destination, destination);
                max_destination = max(max_destination, destination);
                atomicAdd(&d_aid[destination], 1);
            }
            atomicMax(new_right, max_destination);
            atomicMin(new_left, min_destination);
        }
    }
}

__global__ void d_activation(int* d_aid, int* d_apr, bool* d_is_active, int* left, int* right){
    int id = blockDim.x * blockIdx.x + threadIdx.x + *left;
    if(id <= *right){
        // printf("%d: %d, %d\n", id, d_aid[id], d_apr[id]);
    }
    if(id <= *right && d_aid[id] >= d_apr[id]) {
        d_is_active[id] = true;
    }
}

__global__ void d_deactivation(int* d_aid, int* d_apr, bool* d_is_active, int* left, int* right){
    int id = blockDim.x * blockIdx.x + threadIdx.x + *left;
    if(id > *left && id < *right){
        if(d_is_active[id-1] || d_is_active[id+1]){
            ;
        } else {
            d_is_active[id] = false;
        }
    }
}




/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    // int *d_activeVertex;
	// cudaMalloc(&d_activeVertex, L*sizeof(int));


/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/

    // Initialize d_aid
    cudaMemset(d_aid, 0, V * sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
	cudaMalloc(&d_activeVertex, L*sizeof(int));
    cudaMemset(d_activeVertex, 0, L * sizeof(int));

    // Boolean array to track if vertex is active.
    bool *d_is_active;
    cudaMalloc(&d_is_active, V * sizeof(bool));
    cudaMemset(d_is_active, false, V * sizeof(bool));

    // Left of current level
    int* left; 
    cudaMalloc(&left, sizeof(int));
    cudaMemset(left, 0, sizeof(int));
    int* new_left; 
    cudaMalloc(&new_left, sizeof(int));
    cudaMemset(new_left, 0, sizeof(int));
    // Right of current level
    int* right; 
    cudaMalloc(&right, sizeof(int));
    cudaMemset(right, 0, sizeof(int));
    int* new_right; 
    cudaMalloc(&new_right, sizeof(int));
    cudaMemset(new_right, 0, sizeof(int));

    int n_blocks = (V + 511) / 512;

    // Initial processing of level 0.
    // ==============================
    d_level_0<<<n_blocks, 512>>>(V, d_apr, d_is_active, right);
    cudaDeviceSynchronize();

    // Call kernel for each level.
    // ===========================
    int* h_left = (int*)malloc(sizeof(int));
    int* h_right = (int*)malloc(sizeof(int));

    for(int i = 0; i < L; i++){
        n_blocks = 10;

        // Activation
        d_activation<<<n_blocks, 1024>>>(d_aid, d_apr, d_is_active, left, right);
        cudaDeviceSynchronize();

        // deactivation
        d_deactivation<<<n_blocks, 1024>>>(d_aid, d_apr, d_is_active, left, right);
        cudaDeviceSynchronize();

        // Calculate active indegree for level i + 1:
        d_get_aid<<<n_blocks, 1024>>>(V, d_offset, d_csrList, d_aid, d_apr, d_is_active, d_activeVertex, left, new_left, right, new_right, i);
        cudaDeviceSynchronize();

        cudaMemcpy(left, new_left, sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(right, new_right, sizeof(int), cudaMemcpyDeviceToDevice);
    }
    
    cudaMemcpy(h_activeVertex, d_activeVertex, L * sizeof(int), cudaMemcpyDeviceToHost);

/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
