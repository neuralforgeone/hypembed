use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn matmul_bench(c: &mut Criterion) {
    use hypembed::tensor::matmul;
    use hypembed::tensor::{Shape, Tensor};

    let sizes = [(64, 64), (128, 128), (256, 256), (384, 384)];

    for (m, n) in sizes {
        let a = Tensor::full(Shape::new(vec![m, n]), 0.01);
        let b = Tensor::full(Shape::new(vec![n, m]), 0.01);
        c.bench_function(&format!("matmul_{}x{}", m, n), |bench| {
            bench.iter(|| {
                black_box(matmul::matmul(&a, &b).unwrap());
            });
        });
    }
}

fn softmax_bench(c: &mut Criterion) {
    use hypembed::tensor::softmax;
    use hypembed::tensor::{Shape, Tensor};

    let t = Tensor::full(Shape::new(vec![32, 128]), 0.5);
    c.bench_function("softmax_32x128", |bench| {
        bench.iter(|| {
            black_box(softmax::softmax(&t).unwrap());
        });
    });
}

fn layernorm_bench(c: &mut Criterion) {
    use hypembed::tensor::layernorm;
    use hypembed::tensor::{Shape, Tensor};

    let hidden = 384;
    let seq = 128;
    let t = Tensor::full(Shape::new(vec![seq, hidden]), 0.5);
    let gamma = Tensor::ones(Shape::new(vec![hidden]));
    let beta = Tensor::zeros(Shape::new(vec![hidden]));

    c.bench_function("layernorm_128x384", |bench| {
        bench.iter(|| {
            black_box(layernorm::layer_norm(&t, &gamma, &beta, 1e-12).unwrap());
        });
    });
}

fn l2_normalize_bench(c: &mut Criterion) {
    use hypembed::tensor::normalize;
    use hypembed::tensor::{Shape, Tensor};

    let t = Tensor::full(Shape::new(vec![32, 384]), 0.5);
    c.bench_function("l2_normalize_32x384", |bench| {
        bench.iter(|| {
            black_box(normalize::l2_normalize(&t, 1e-12).unwrap());
        });
    });
}

criterion_group!(
    benches,
    matmul_bench,
    softmax_bench,
    layernorm_bench,
    l2_normalize_bench
);
criterion_main!(benches);
