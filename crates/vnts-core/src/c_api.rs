//! C-FFI exports for seamless Python and native C bindings.

use crate::cholesky::Cholesky;
use crate::matrix::Matrix;
use crate::nullspace::NullspaceEngine;
use crate::woodbury::Woodbury;

#[no_mangle]
pub extern "C" fn vnts_version_major() -> u32 {
    0
}

#[no_mangle]
pub extern "C" fn vnts_version_minor() -> u32 {
    9
}

/// Solves Ridge regression in pure native Rust:
/// X: (N x D), Y: (N x K), out_w: (D x K)
#[no_mangle]
pub unsafe extern "C" fn vnts_native_solve_ridge(
    x_ptr: *const f32,
    y_ptr: *const f32,
    n: usize,
    d: usize,
    k: usize,
    reg_lambda: f32,
    out_w_ptr: *mut f32,
) -> i32 {
    if x_ptr.is_null() || y_ptr.is_null() || out_w_ptr.is_null() {
        return -1;
    }

    let x_slice = std::slice::from_raw_parts(x_ptr, n * d);
    let y_slice = std::slice::from_raw_parts(y_ptr, n * k);

    let x_mat = Matrix::new(n, d, x_slice.to_vec());
    let y_mat = Matrix::new(n, k, y_slice.to_vec());

    let w_mat = Cholesky::solve_ridge(&x_mat, &y_mat, reg_lambda);

    std::ptr::copy_nonoverlapping(w_mat.data.as_ptr(), out_w_ptr, d * k);
    0
}

/// Solves Nullspace-Projected Ridge update:
/// X: (N x D), Y: (N x K), U: (D x r), out_dw: (D x K)
#[no_mangle]
pub unsafe extern "C" fn vnts_native_solve_nullspace(
    x_ptr: *const f32,
    y_ptr: *const f32,
    u_ptr: *const f32,
    n: usize,
    d: usize,
    k: usize,
    r: usize,
    reg_lambda: f32,
    out_dw_ptr: *mut f32,
) -> i32 {
    if x_ptr.is_null() || y_ptr.is_null() || out_dw_ptr.is_null() {
        return -1;
    }

    let x_slice = std::slice::from_raw_parts(x_ptr, n * d);
    let y_slice = std::slice::from_raw_parts(y_ptr, n * k);
    let x_mat = Matrix::new(n, d, x_slice.to_vec());
    let y_mat = Matrix::new(n, k, y_slice.to_vec());

    let u_mat = if !u_ptr.is_null() && r > 0 {
        let u_slice = std::slice::from_raw_parts(u_ptr, d * r);
        Some(Matrix::new(d, r, u_slice.to_vec()))
    } else {
        None
    };

    let dw_mat = NullspaceEngine::solve_nullspace_projected(
        &x_mat,
        &y_mat,
        u_mat.as_ref(),
        reg_lambda,
    );

    std::ptr::copy_nonoverlapping(dw_mat.data.as_ptr(), out_dw_ptr, d * k);
    0
}

/// Executes an online rank-1 Woodbury update:
/// P: (D x D), x: (D,), out_p: (D x D), out_k: (D,)
#[no_mangle]
pub unsafe extern "C" fn vnts_native_woodbury_step(
    p_ptr: *const f32,
    x_ptr: *const f32,
    d: usize,
    out_p_ptr: *mut f32,
    out_k_ptr: *mut f32,
) -> i32 {
    if p_ptr.is_null() || x_ptr.is_null() || out_p_ptr.is_null() || out_k_ptr.is_null() {
        return -1;
    }

    let p_slice = std::slice::from_raw_parts(p_ptr, d * d);
    let x_slice = std::slice::from_raw_parts(x_ptr, d);

    let p_mat = Matrix::new(d, d, p_slice.to_vec());
    let (p_next, k_vec) = Woodbury::update(&p_mat, x_slice);

    std::ptr::copy_nonoverlapping(p_next.data.as_ptr(), out_p_ptr, d * d);
    std::ptr::copy_nonoverlapping(k_vec.as_ptr(), out_k_ptr, d);
    0
}
