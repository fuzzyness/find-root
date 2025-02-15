/// Defines the function f(x) = x⁴ - 3x² - 3.
///
/// # Arguments
///
/// * `x` - The input value.
///
/// # Returns
///
/// The computed value of f(x).
fn f(x: f64) -> f64 {
    x.powi(4) - 3.0 * x.powi(2) - 3.0
}

/// Defines the function g(x) such that g(x) = (3x² + 3)¹/⁴.
///
/// This is used for the fixed-point iteration method (i.e., a fixed point satisfies x = g(x)).
///
/// # Arguments
///
/// * `x` - The input value.
///
/// # Returns
///
/// The computed value of g(x).
fn g(x: f64) -> f64 {
    (3.0 * x * x + 3.0).powf(1.0 / 4.0)
}

/// Defines the prime of f(x) such that f'(x) = 4x³- 6x.
///
/// # Arguments
///
/// * `x` - The input value.
///
/// # Returns
///
/// The computed value of f'(x).
fn f_prime(x: f64) -> f64 {
    4.0 * x.powi(3) - 6.0 * x
}

/// Performs the bisection method to find a root of the function f(x)
/// within the interval [a, b] to within the specified tolerance.
///
/// The method repeatedly bisects the interval and selects the subinterval
/// in which the sign of f(x) changes, ensuring that a root is contained within.
///
/// # Arguments
///
/// * `f` - The function where the root is sought. It must have opposite signs at `left` and `right`.
/// * `left` - The left endpoint of the interval.
/// * `right` - The right endpoint of the interval.
/// * `tolerance` - The tolerance of convergence.
///
/// # Returns
///
/// The approximate root of f(x) = 0 within the interval [left, right].
///
/// # Panics
///
/// Panics if f(left) and f(right) do not have opposite signs.
fn bisection(f: fn(f64) -> f64, mut left: f64, mut right: f64, tolerance: f64) -> f64 {
    // Ensure that the function has opposite signs at the endpoints.
    // If f(a) * f(b) < 0, then by IVT, there is at least one root in [a, b].
    assert!(f(right) * f(left) < 0.0, "f(a) and f(b) must have opposite signs");

    // Iterate until the interval's half-length is less than the tolerance.
    while (right - left) / 2.0 > tolerance {
        // Calculate the midpoint of the current interval.
        let mid = (left + right) / 2.0;

        // If the f(c) is less than the tolerance, we consider it to be the root.
        if f(mid).abs() < tolerance {
            return mid;
        }

        // If f(a) and f(c) have opposite signs, then the root lies in [a, c].
        // Otherwise, it lies in [c, b].
        if f(left) * f(mid) < 0.0 {
            right = mid;
        } else {
            left = mid;
        }
    }

    // The interval [a, b] is less than the tolerance so the midpoint approximates the root.
    (right + left) / 2.0
}

/// Performs fixed-point iteration to find a solution to x = g(x).
///
/// The method repeatedly applies the function g(x) to the current approximation
/// until the change between successive approximations is less than the specified tolerance.
///
/// # Arguments
///
/// * `g` - The iteration function used to compute successive approximations.
/// * `current` - The initial guess and current value for the fixed point.
/// * `tolerance` - The tolerance for convergence.
///
/// # Returns
///
/// The approximate fixed-point that satisfies x = g(x) within the given tolerance.
fn fixed_point(g: fn(f64) -> f64, mut current: f64, tolerance: f64) -> f64 {
    // Iterate until the change is smaller than the tolerance
    loop {
        // Compute the next iteration value.
        let next = g(current);

        // If the difference is within the tolerance, return the current approximation.
        if (next - current).abs() < tolerance {
            return next;
        }

        // Update the current guess.
        current = next;
    }
}

/// Performs Newton's method to approximate a root of the function f(x).
///
/// The method uses the iterative formula:
///   x₁ = x₀ - f(x₀)/f'(x₀)
/// starting from an initial guess, and continuing until the difference between
/// successive approximations is less than the specified tolerance.
///
/// # Arguments
///
/// * `f` - The function where the root is sought.
/// * `f_prime` - The derivative of the function f(x).
/// * `start` - The initial guess for the root.
/// * `tolerance` - The tolerance for cconvergence.
///
/// # Returns
///
/// * `Some(root)` if the method converges to an approximation of the root.
/// * `None` if the derivative is too close to zero (risking division by zero).
fn newton(
    f: impl Fn(f64) -> f64,
    f_prime: impl Fn(f64) -> f64,
    start: f64,
    tolerance: f64,
) -> Option<f64> {
    // Set starting point to the initial guess.
    let mut current = start;

    loop {
        // Evaluate the function and derivative at the current approximation.
        let f_current = f(current);
        let fp_current = f_prime(current);

        // Check if the derivative is too small to avoid division by zero.
        if fp_current.abs() < std::f64::EPSILON {
            return None;
        }

        // Compute the next approximation using Newton's formula: xₙₑₓₜ = x - f(x)/f'(x)
        let next = current - f_current / fp_current;

        // Check if the difference between successive approximations is within tolerance.
        if (next - current).abs() < tolerance {
            return Some(next);
        }

        // Update the current approximation for the next iteration.
        current = next;
    }
}

/// Performs the secant method to approximate a root of the function f(x).
///
/// The method uses two initial guesses to approximate a root by computing
/// the intersection of the secant line through the points (xₙ₋₁, f(xₙ₋₁))
/// and (xₙ, f(xₙ)) with respect to the x-axis.
///
/// # Arguments
///
/// * `f` - The function where the root is sought.
/// * `previous` - The previous approximation value (first initial guess).
/// * `current` - The current approximation value (second initial guess).
/// * `tolerance` - The tolerance for convergence. The iteration stops when
///                 the absolute difference between successive approximations
///                 is less than the tolerance.
///
/// # Returns
///
/// * `Some(root)` if the method converges to an approximation of the root.
/// * `None` if the method is too close to zero (risking division by zero).
fn secant(f: impl Fn(f64) -> f64, mut previous: f64, mut current: f64, tolerance: f64) -> Option<f64> {
    loop {
        // Evaluate the function at the current approximation.
        let f_previous = f(previous);
        let f_current = f(current);

        // Check if the derivative is too small to avoid division by zero.
        if (f_current - f_previous).abs() < std::f64::EPSILON {
            return None;
        }

        // Compute the next approximation using the secant formula:
        // x₂ = x₁ - f(x₁) * (x₁ - x₀) / (f(x₁) - f(x₀))
        let next = current - f_current * (current - previous) / (f_current - f_previous);


        // Check if the difference between successive approximations is within tolerance.
        if (next - current).abs() < tolerance {
            return Some(next);
        }

        // Update the previous two approximations for the next iteration.
        previous = current;
        current = next;
    }
}

/// Entry point of the program.
///
/// Approximates a root of the equation f(x) = 0 in the interval [a, b] using:
///   - The bisection method
///   - The fixed-point iteration method
///   - Newton's method
///   - The secant method
///
/// It then prints the approximated roots for each method.
fn main() {
    // Define the tolerance level for the approximation.
    let tolerance = 1e-6;
    // Define the endpoints of the interval [a, b].
    let a = 1.0;
    let b = 2.0;
    // Define the initial guess for the approximation.
    let initial_guess = 2.0;

    // ---------------------------
    // Bisection Method
    // ---------------------------
    // Approximate the root using the bisection method.
    let bisection = bisection(f, a, b, tolerance);

    // Print approximate root for the bisection method to 7 decimal places.
    println!("Approximate root using the bisection method: {:.7}", bisection);

    // ---------------------------
    // Fixed-Point Iteration
    // ---------------------------
    // Approximate the root using the fixed-point iteration method.
    let fixed_point = fixed_point(g, initial_guess, tolerance);

    // Print approximate root for the fixed-point iteration method to 7 decimal places.
    println!("Approximate root using the fixed-point iteration method: {:.7}", fixed_point);

    // ---------------------------
    // Newton's Method
    // ---------------------------
    // Print approximate root for Newton's method to 7 decimal places.
    match newton(f, f_prime, initial_guess, tolerance) {
        Some(root) => println!("Approximate root using Newton's method: {:.7}", root),
        None => println!("Newton's method did not converge."),
    }

    // ---------------------------
    // Secant Method
    // ---------------------------
    // Print approximate root for the secant method to 7 decimal places.
    match secant(f, a, b, tolerance) {
        Some(root) => println!("Approximate root using the secant method: {:.7}", root),
        None => println!("Secant method did not converge."),
    }
}
