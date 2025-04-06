extern "C"
__global__ void tensorAddition(int n, float *A, float *B, float *C) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) 
        C[i] = A[i] + B[i];
}

extern "C"
__global__ void tensorSubtraction(int n, float *A, float *B, float *C) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) 
        C[i] = A[i] - B[i];
}

extern "C"
__global__ void tensorMultiplication(int n, float *A, float *B, float *C) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) 
        C[i] = A[i] * B[i];
}

extern "C"
__global__ void tensorDivision(int n, float *A, float *B, float *C) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) 
        C[i] = A[i] / B[i];
}

// scalar operations

extern "C"
__global__ void tensorSAddition(int n, float *A, float b, float *C) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) 
        C[i] = A[i] + b;
}

extern "C"
__global__ void tensorSSubtraction(int n, float *A, float b, float *C) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) 
        C[i] = A[i] - b;
}

extern "C"
__global__ void tensorSMultiplication(int n, float *A, float b, float *C) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) 
        C[i] = A[i] * b;
}

extern "C"
__global__ void tensorSDivision(int n, float *A, float b, float *C) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) 
        C[i] = A[i] / b;
}