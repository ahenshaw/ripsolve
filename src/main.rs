#![feature(nll)]
#[macro_use(s,stack)]
extern crate ndarray;
extern crate rand;

use ndarray::prelude::*;
use ndarray::Zip;
use ndarray::Array;
use rand::{Rng,thread_rng};
use std::time::Instant;

type Matrix = Array2<f32>;
type Vector = Array1<f32>;

#[derive(Debug, PartialEq, Default)]
struct Pivot {
    row: usize,
    col: usize
}

#[allow(non_snake_case)]
fn initial_tableau(c:&Vector, A:&Matrix, b:&Vector) -> Matrix {
    let mut tableau = Array::from_elem((A.rows()+1, A.cols()+1), 0.);
    
    tableau.slice_mut(s![..-1,..-1]).assign(A);
    tableau.slice_mut(s![..-1,  -1]).assign(b);
    tableau.slice_mut(s![-1,  ..-1]).assign(c);

    tableau
}

fn can_improve(tableau:&Matrix) -> bool {
    tableau.slice(s![-1, ..-1]).iter().any(|&x| x < 0.0)
}

fn find_pivot(tableau:&Matrix) -> Result<Pivot, &'static str>{
    // pick minimum index of most negative value in the last row
    let last_row = tableau.slice(s![-1, ..-1]);
    let mut vmax =  0.0;
    let mut col  =  std::usize::MAX;
    for (i, &value) in last_row.iter().enumerate() {
        if value < vmax {
            col = i;
            vmax = value;
        }
    }

    // pick row index minimizing the quotient
    let lc = tableau.cols() - 1;
    let mut qmax =  std::f32::INFINITY;
    let mut row = std::usize::MAX;
    for (i, r) in tableau.slice(s![..-1, ..]).outer_iter().enumerate() {
        if r[col] > 0.0 {
            let q = r[lc] / r[col];
            if qmax > q {
                qmax = q;
                row = i;
            }
        }
    }
    // if row is still std::usize::MAX, then solution is unbounded
    if row == std::usize::MAX {
        Err("Solution is unbounded")
    } else {
        Ok(Pivot{row:row, col:col})
    }
}

fn do_pivot(tableau:&mut Matrix, pivot:&Pivot) {
    let mut pivoting_row = tableau.subview_mut(Axis(0), pivot.row);
    pivoting_row /= pivoting_row[pivot.col];

    let pivoting_row = pivoting_row.to_owned(); 
    
    // subtract the pivot row from all other rows
    for (i, mut r) in tableau.outer_iter_mut().enumerate() {
        if i != pivot.row {
            let multiplier = r[pivot.col];
            Zip::from(&mut r)
                 .and(&pivoting_row)
                 .apply(|w, &x| {*w -= multiplier*x});
        }
    }
}

#[allow(non_snake_case)]
fn simplex(c:&Vector, A:&Matrix, b:&Vector, basis:&mut [usize]) -> Result<f32, &'static str>  {
    let mut tableau = initial_tableau(c, A, b);
    let mut count = 0;
    while can_improve(&tableau) {
        count += 1;
        // println!("Basis = {:?}\n", basis);
        let pivot = find_pivot(&tableau)?;
        // println!("{:?}", pivot);
        do_pivot(&mut tableau, &pivot);
        // println!("{}", tableau);
        basis[pivot.row] = pivot.col;
    }
    println!{"Iterations: {}", count};
    let lc = tableau.cols() - 1;
    let lr = tableau.rows() - 1;
    let last_col = tableau.subview_mut(Axis(1), lc);
    let cost = -last_col[lr];
    Ok(cost)
}

#[allow(non_snake_case)]
fn main() {
    let vars = 100; 
    let constraints = 50;
    let mut rng = thread_rng();
    // let c = array![-10., -12., -12., 0., 0., 0.];
    // let A = array![[1.,    2.,   2., 1., 0., 0.],
    //                [2.,    1.,   2., 0., 1., 0.],
    //                [2.,    2.,   1., 0., 0., 1.]];
    //let b = array![20., 20., 20.];
    

    // let c = array![-16.0, -12.0, -18.0, 0., 0., 0.];
    // let A = array![[2.,    2.,   1., 1., 0., 0.],
    //                [2.,    1.,   2., 0., 1., 0.],
    //                [1.,    2.,   1., 0., 0., 1.]];

    let b = Vector::zeros(constraints)+10.0;
    let mut a_p = Matrix::zeros((constraints, vars));
    for x in a_p.iter_mut() {
        *x = if rng.gen::<f32>() < 0.5 {1.0} else {2.0};
    }
    let A = stack![Axis(1), a_p, Matrix::eye(constraints)];

    let mut c_p = Vector::zeros(vars);
    for x in c_p.iter_mut() {
        *x = rng.gen_range::<f32>(-20., -10.);
    }
    let c = stack![Axis(0), c_p, Vector::zeros(constraints)];


    //println!("{}\n{}", A, c);
    let mut basis:Vec<usize> = (vars..vars+constraints).collect();

    let now = Instant::now();
    match simplex(&c, &A, &b, &mut basis) {
        Ok(cost) => println!("Cost is {}", cost),
        Err(e) => println!("{}", e)
    }
    println!("Elapsed time: {:?}", now.elapsed());
}
