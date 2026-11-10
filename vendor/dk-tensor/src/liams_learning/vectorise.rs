use multiversion::multiversion;

//! THIS IS AN INSTRUCTIONAL FILE WHERE I AM EXPLORING OPTIMAL CPU ALLOCATION & TILE CACHING


// We assume you've already extracted flat &[f32] slices 
// and checked that `inner == 1`.
#[multiversion(targets(
    "x86_64+avx2+fma",
    "x86_64+sse4.1",
    "aarch64+neon"
))]
pub fn log_softmax_last_dim_kernel(input: &[f32], output: &mut [f32], len: usize) {
    // Process row by row. This perfectly aligns with cache lines.
    let in_rows = input.chunks_exact(len);
    let out_rows = output.chunks_exact_mut(len);

    for (in_row, out_row) in in_rows.zip(out_rows) {
        
        // 1. Find Max (LLVM can auto-vectorize horizontal reductions)
        let mut max = f32::NEG_INFINITY;
        for &x in in_row {
            if x > max {
                max = x;
            }
        }

        // 2. Sum of Exps
        let mut sum_exp = 0.0;
        for &x in in_row {
            sum_exp += (x - max).exp();
        }

        // 3. Final Log Softmax
        let lse = max + sum_exp.ln();
        for (&x, o) in in_row.iter().zip(out_row.iter_mut()) {
            *o = x - lse;
        }
    }
}

/// Tiled matrix multiplication: [m, k] x [k, n] -> [m, n]
/// Assumes a, b, and c are flat, contiguous slices.
pub fn mm_tiled(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    // 32x32 f64s = 1024 elements = 8KB. 
    // Three of these fit comfortably in a standard 32KB L1 cache.
    let block_size = 32; 

    // 1. Iterate over the blocks
    for i_block in (0..m).step_by(block_size) {
        for l_block in (0..k).step_by(block_size) {
            for j_block in (0..n).step_by(block_size) {
                
                // Calculate the boundaries of the current block
                let i_max = (i_block + block_size).min(m);
                let l_max = (l_block + block_size).min(k);
                let j_max = (j_block + block_size).min(n);
                
                // 2. Compute the matrix multiplication FOR THIS BLOCK ONLY
                // Notice the loop order: i, then l, then j. 
                // This order ensures we iterate through 'c' and 'b' contiguously.
                for i in i_block..i_max {
                    for l in l_block..l_max {
                        let a_val = a[i * k + l];
                        
                        for j in j_block..j_max {
                            c[i * n + j] += a_val * b[l * n + j];
                        }
                    }
                }
                
            }
        }
    }
}