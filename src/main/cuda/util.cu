extern "C"
__global__ void cat(int m, float *x, float *y, int xSize, int ySize, float *output){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m){
			int side = i % (xSize + ySize);
			int index = i / (xSize + ySize);
			if (side < xSize)
					output[i] = x[index * xSize + side];
			else
					output[i] = y[index * ySize + side - xSize];
	}
}

// m, n - shape of input
extern "C"
__global__ void matrixTransposition(int m, int n, float *input, float *output){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < n && j < m)
		output[i * m + j] = input[j * n + i];
}

extern "C"
__global__ void broadcasting(int elems2copy, int copyCount, int n, float* input, float* output){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		int blockIdx = i / elems2copy / copyCount;
		int valueIdx = i % elems2copy;
		output[i] = input[blockIdx * elems2copy + valueIdx];
	}
}


extern "C"
__global__ void indexSelection(int s, int a, int f, int n, float *input, float *output){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n){
		int n = i / s;
		int r = i % s;
		int index = s * (a * n + f) + r;
		
		output[i] = input[index];
	}
}