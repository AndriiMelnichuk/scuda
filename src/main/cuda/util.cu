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
	if(i < n * m){
		int IdxI = i % n;
		int IdxJ = i / n;
		output[IdxI * m + IdxJ] = input[n * IdxJ + IdxI];
	}
		
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


extern "C"
__global__ void createFeatureForConv(
	int C, int H, int W, 
	int lhiX, int lhiY, int imgN, 
	int kernelSize, float *input, float *output)
	{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < kernelSize * kernelSize * C){
		int c = i / kernelSize / kernelSize;
		int h = lhiY + i / kernelSize % kernelSize;
		int w = lhiX + i % kernelSize;
		
		float res = 0;
		if (!(h >= H || h < 0 || w >= W || w < 0))
			res = input[C * H * W * imgN + H * W * c + W * h + w];
		output[i] = res;
	}
}