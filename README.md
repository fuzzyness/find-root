# Root-finding methods

Approximate the root of a given function using several root finding methods.

## Description

1.  Write a program to find an approximation to the solution to the equation f(x) = 0 lying in [a, b] that is accurate to within 10⁻⁶, using:
    
    1.  The bisection method
    2.  The fixed-point iteration method
    3.  The Newton's method
    4.  The secant method
    
    where f(x) = x⁴ - 3x² - 3 and [a, b] = [1, 2].

2.  For each method, sketch the following errors with respect to the number n of iterations:
    
    -   **Error 1:** |pₙ - pₙ₋₁| / |pₙ|
    -   **Error 2:** |pₙ - pₙ₋₁|
    -   **Error 3:** |f(pₙ)|
    
    where (pₙ) is the sequence generated by each of these methods.

3.  Use these graphs to compare the three methods.

## Installation

Make sure you have [Rust](https://doc.rust-lang.org/book/ch01-01-installation.html) already installed, then:

    git clone https://github.com/fuzzyness/findroot.git

Navigate to the root of the project, then run the program:

    cargo run

Running the program should result in an output similar to the following:

    Approximate root using the bisection method: 1.9471235
    Approximate root using the fixed-point iteration method: 1.9471233
    Approximate root using Newton's method: 1.9471230
    Approximate root using the secant method: 1.9471230

A new directory should also have been created in the root of the program called `images`. This is where the graphs generated by the program will be stored.

A written comparison of the root-finding methods can be found in `comparison.md`.

Documentation for this program can be generated by running:

    cargo rustdoc

The newly generated documentation can be found in `target/doc/findroot/index.html`.

