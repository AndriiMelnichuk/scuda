
// n - elements in the array
// elemsInBlock - number of elements in array per block
// input - input array
// output - output array
// thread in block must be less than elemsInBlock
extern "C"
__global__ void reduceSum(int n, float *input, float *output) {
    extern __shared__ float sdata[];
    int elemsInBlock = n / gridDim.x;
    int elemsPerThread = elemsInBlock / blockDim.x;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int from = i * elemsPerThread;
    int to = from + elemsPerThread - 1;
    if (tid == blockDim.x - 1 && blockIdx.x == gridDim.x - 1)
        to = n - 1;
    
    sdata[tid] = 0;
    for (int j = from; j < to + 1; j++) 
        sdata[tid] += input[j];
    __syncthreads();

    if (tid == 0){
        output[blockIdx.x] = 0; 
        for (int j = 0; j < blockDim.x; j++) 
            output[blockIdx.x] += sdata[j];
    }
}
