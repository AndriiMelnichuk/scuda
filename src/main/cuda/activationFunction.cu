
extern "C"
__global__ void sigmoid(int n, float *input, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

extern "C"
__global__ void ReLU(int n, float *input, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = max(0.0f, input[i]);
    }
}

extern "C"
__global__ void stableSoftmax(int m, int n, float *input, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        float maxX = input[i * n];
        for (int j = 1; j != n; j++)
            if (input[i * n + j] > maxX)
                maxX = input[i * n + j];
        
        float eSum = 0;
        for (int j = 0; j != n; j++)
            eSum += expf(input[i * n + j] - maxX);
        
        for (int j = 0; j < n; ++j)
            output[i * n + j] = expf(input[i * n + j] - maxX) / eSum;
    }
}


extern "C"
__global__ void stableSoftmaxGrad(int m, int n, float *softmax, float *chainGrad, float *result){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n){
        float res = 0;
        for(int k = 0; k != n; k++){
            float p = 0;
            if (k == j)
                p = softmax[i * n + j] * (1 - softmax[i * n + j]);
            else
                p = -softmax[i * n + j] * softmax[i * n + k];
            res += p * chainGrad[i * n + k];
        }
        result[i * n + j] = res;
    }
}