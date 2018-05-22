#define SWAP(a,b) {__local int * tmp=a; a=b; b=tmp;}

__kernel void gpu_scan(__global int* input, __global int* output, __local int* a, __local int* b, __global int* c, int n) {
	uint gid = get_global_id(0);
	uint lid = get_local_id(0);
	uint block_size = get_local_size(0);
	uint threads_size = get_global_size(0);

	c[gid] = input[gid];
	barrier(CLK_GLOBAL_MEM_FENCE);

	int offset = 0;
	int c_offset = 0;
	int deep = 0;
	int max_step = 0;
	for (uint step = 1; step < n; step *= block_size) {
		a[offset + lid] = b[offset + lid] = c[c_offset + gid];
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint s = 1; s < block_size; s <<= 1) {
			if (step * gid < n) {
				if (lid >(s - 1)) {
					b[offset + lid] = a[offset + lid] + a[offset + lid - s];
				}
				else {
					b[offset + lid] = a[offset + lid];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			SWAP(a, b);
		}

		c_offset += threads_size / step;

		if (lid == 0) {
			if (step * (gid + block_size - 1) < n) {
				c[c_offset + gid / block_size] = a[offset + block_size - 1];
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);

		offset += block_size;
		deep += 1;
		max_step = step;
	}

	for (uint step = max_step; step > 0; step /= block_size) {
		offset -= block_size;
		deep -= 1;

		if (gid >= block_size) {
			if (step * gid / block_size < n) {
				a[offset + lid] += c[c_offset + gid / block_size - 1];
			}
		}
		c_offset -= threads_size / step;
		if (step * gid < n) {
			c[c_offset + gid] = a[offset + lid];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	output[gid] = a[lid];
}