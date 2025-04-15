// cross entropy loss
extern "C"
__global__ void crossEntropyLoss(int m, int n, float *input, float *target, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        int targetIndex = (int)target[i];
        float pr = input[i * n + targetIndex];   
        output[i] = -logf(pr);
    }
}

extern "C"
__global__ void crossEntropyLossGrad(int m, int n, float *input, float *target, float *chainGrad, float *output){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m){
        for(int j=0; j!=n; j++)
            output[i * n + j] = 0;
        int j = (int)target[i];
        output[i * n + j] = chainGrad[i] / input[i * n + j];
    }
}