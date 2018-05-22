__kernel void gpu_convolution(__global double* a, __global double* b, __global double* c, int n, int m) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (i >= n || j >= n) {
		return;
	}

	double value = 0;
	int hm = (m - 1) / 2;
	for (int k = -hm; k <= hm; k++) {
		if (i + k < 0 || i + k >= n) {
			continue;
		}
		for (int l = -hm; l <= hm; l++) {
			if (j + l < 0 || j + l >= n) {
				continue;
			}
			value += a[(i + k) * n + j + l] * b[(k + hm) * m + (l + hm)];
		}
	}
	c[i * n + j] = value;
	barrier(CLK_GLOBAL_MEM_FENCE);
}