#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

// Structure to store information about requests.
typedef struct {
    int req_id;
    int req_cen;
    int req_fac;
    int req_start;
    int req_slots;
}request_str;

// Global variable in GPU
__device__ int d_N;
__device__ int d_R;


// Compare  function to sort the requests.

int comp(const void* x_ptr, const void* y_ptr){
    request_str x = *((request_str*) x_ptr);;
    request_str y = *((request_str*) y_ptr);;

    if(x.req_cen != y.req_cen) return x.req_cen - y.req_cen;
    if(x.req_fac != y.req_fac) return x.req_fac - y.req_fac;
    if(x.req_id != y.req_id) return x.req_id - y.req_id;
    return 0;
}


//*******************************************

// Write down the kernels here

//***********************************************

__global__ void set_vars(int N, int R){
    d_N = N;
    d_R = R;
}

__global__ void get_req_per_fac(int* d_req_per_fac, int* d_facilty_presum, request_str* d_requests, int R){
    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    int fid;
    if(tid < R){
        int center_id = d_requests[tid].req_cen;
        int fac_id = d_requests[tid].req_fac;
        fid = d_facilty_presum[center_id] + fac_id;
        atomicAdd(&d_req_per_fac[fid], 1);
    }
}

__global__ void print_debug(int* arr, int sz){
    for(int i = 0; i < sz; i++){
        printf("%d:  %d\n", i, arr[i]);
    }
}


__global__ void solve(int* d_req_pre_fac_presum, int* d_capacity, request_str* d_requests, int* d_succ_reqs, int* d_tot_reqs, int* d_success, int* d_fail, int k1){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int slot_capacity[25];
    for(int i = 0; i < 25; i++){
        slot_capacity[i] = d_capacity[tid];
    }
    if(tid < k1){
        int start = d_req_pre_fac_presum[tid];
        int end = d_req_pre_fac_presum[tid+1];
        for(int i = start; i < end; i++){
            int req_id = d_requests[i].req_id;        
            int req_cen = d_requests[i].req_cen;      
            int req_fac = d_requests[i].req_fac;      
            int req_start = d_requests[i].req_start;  
            int req_slots = d_requests[i].req_slots;  
            int available = 1;
            for(int j = req_start; j < (req_start + req_slots); j++){
                available *= slot_capacity[j] > 0 ? 1 : 0;
            }
            if(available == 1){
                for(int j = req_start; j < (req_start + req_slots); j++){
                    slot_capacity[j]--;
                }
                atomicAdd(&d_succ_reqs[req_cen], 1);
                atomicAdd(d_success, 1);
            } else {
                atomicAdd(d_fail, 1);
            }
            atomicAdd(&d_tot_reqs[req_cen], 1);
            __syncthreads();
            // if(req_cen == 1){
            //     printf("%d %d %d %d\n", req_id, available, d_succ_reqs[req_cen], d_tot_reqs[req_cen]);
            // }
        }
    }
}



int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 


    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    int k1=0 , k2 = 0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      }
    }
    

    /***************** Student code: start*************************/

    // Get prefix sum of facilities per centre in host.
    int* facility_presum = (int*) malloc((k1 + 1) * sizeof(int));
    facility_presum[0] = 0;
    for(int i = 1; i <= N; i++){
        facility_presum[i] = facility_presum[i-1] + facility[i-1];
    }

    // Copy prefix sum of facilities to device.
    int* d_facility_presum;
    cudaMalloc(&d_facility_presum, (k1 + 1)* sizeof(int));
    cudaMemcpy(d_facility_presum, facility_presum, (k1 + 1)* sizeof(int), cudaMemcpyHostToDevice);

    /****************** Student code: end *************************/

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request

    // To store requsts as an array of structures.
    request_str *requests;
    
    // Allocate memory on CPU 
	int R;
	fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request
    requests = (request_str*) malloc((R) *sizeof (request_str)); // Requests in array of structures.

    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
       requests[j].req_id = req_id[j];
       requests[j].req_cen = req_cen[j];
       requests[j].req_fac = req_fac[j];
       requests[j].req_start = req_start[j];
       requests[j].req_slots = req_slots[j];
    }


    /***************** Student code: start*************************/

    qsort(requests, R, sizeof(request_str), comp);

    /****************** Student code: end *************************/
		
    //*********************************
    // Call the kernels here
    //********************************

    // Copy values of N and R to device.
    set_vars<<<1, 1>>>(N, R);

    // Get requests per facility.

    // Copy requests
    request_str* d_requests;
    cudaMalloc(&d_requests, R * sizeof(request_str));
    cudaMemcpy(d_requests, requests, R * sizeof(request_str), cudaMemcpyHostToDevice);

    // requests per facility.
    int* d_req_per_fac;
    cudaMalloc(&d_req_per_fac, k1 * sizeof(int));
    cudaMemset(d_req_per_fac, 0, k1 * sizeof(int));

    int n_block = (R + 1) / BLOCKSIZE + 1;
    get_req_per_fac<<<n_block, BLOCKSIZE>>>(d_req_per_fac, d_facility_presum, d_requests, R);

    // Copy this to host
    int* req_per_fac = (int*) malloc(k1 * sizeof(int));
    cudaMemcpy(req_per_fac, d_req_per_fac, k1 * sizeof(int), cudaMemcpyDeviceToHost);

    // Prefix sum of requests per facility.
    int* req_per_fac_presum = (int*) malloc((k1 + 1) * sizeof(int));
    req_per_fac_presum[0] = 0;
    for(int i = 1; i <= k1; i++){
        req_per_fac_presum[i] = req_per_fac_presum[i-1] + req_per_fac[i-1];
    }

    // Copy to device.
    int* d_req_per_fac_presum;
    cudaMalloc(&d_req_per_fac_presum, (k1 + 1) * sizeof(int));
    cudaMemcpy(d_req_per_fac_presum, req_per_fac_presum, (k1 + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Copy capacity to device
    int* d_capacity;
    cudaMalloc(&d_capacity, k1 * sizeof(int));
    cudaMemcpy(d_capacity, capacity, k1 * sizeof(int), cudaMemcpyHostToDevice);

    // Solve
    int* d_fac_succ;
    cudaMalloc(&d_fac_succ, k1 * sizeof(int));
    cudaMemset(d_fac_succ, 0, k1 * sizeof(int));

    int* d_succ_reqs;
    cudaMalloc(&d_succ_reqs, N * sizeof(int));
    cudaMemset(d_succ_reqs, 0, N * sizeof(int));

    int* d_tot_reqs;
    cudaMalloc(&d_tot_reqs, N * sizeof(int));
    cudaMemset(d_tot_reqs, 0, N * sizeof(int));

    int* d_success;
    cudaMalloc(&d_success, sizeof(int));
    cudaMemset(d_success, 0, sizeof(int));

    int* d_fail;
    cudaMalloc(&d_fail, sizeof(int));
    cudaMemset(d_fail, 0, sizeof(int));

    // Call kernel to solve.
    n_block = (R + 1) / BLOCKSIZE + 1;
    solve<<<n_block, BLOCKSIZE>>>(d_req_per_fac_presum, d_capacity, d_requests, d_succ_reqs, d_tot_reqs, d_success, d_fail, k1);

    cudaMemcpy(succ_reqs, d_succ_reqs, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(tot_reqs, d_tot_reqs, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(&success, d_success, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);

    // Output
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}