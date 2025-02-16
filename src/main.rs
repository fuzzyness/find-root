use std::fs;
use plotters::prelude::{
    BitMapBackend,
    ChartBuilder,
    IntoDrawingArea,
    LineSeries,
    PathElement,
    BLACK,
    BLUE,
    GREEN,
    RED,
    WHITE,
};

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
/// within the interval [a, b] to within the specified tolerance and
/// records the approximations at each iteration.
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
/// A tuple containing the approximate root of f(x) = 0 within the
/// interval [left, right] and a vector containing all the approximations.
///
/// # Panics
///
/// Panics if f(left) and f(right) do not have opposite signs.
fn bisection(f: fn(f64) -> f64, mut left: f64, mut right: f64, tolerance: f64) -> (f64, Vec<f64>) {
    // Ensure that the function has opposite signs at the endpoints.
    // If f(a) * f(b) < 0, then by IVT, there is at least one root in [a, b].
    assert!(f(right) * f(left) < 0.0, "f(a) and f(b) must have opposite signs");

    // Define a vector for storing approximations.
    let mut approximations = Vec::new();

    // Iterate until the interval's half-length is less than the tolerance.
    while (right - left) / 2.0 > tolerance {
        // Calculate the midpoint of the current interval.
        let mid = (left + right) / 2.0;
        // Add the calculated midpoint to the list of approximations.
        approximations.push(mid);

        // If the f(c) is less than the tolerance, we consider it to be the root.
        if f(mid).abs() < tolerance {
            return (mid, approximations);
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
    let result = (right + left) / 2.0;
    approximations.push(result);

    // Return the approximated root and list of approximations.
    (result, approximations)
}

/// Performs fixed-point iteration to find a solution to x = g(x)
/// and records the approximations at each iteration.
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
/// A tuple containing the approximate fixed-point that satisfies x = g(x)
/// within the given tolerance and a vector containing all the approximations.
fn fixed_point(g: fn(f64) -> f64, mut current: f64, tolerance: f64) -> (f64, Vec<f64>) {
    // Define a vector for storing approximations.
    let mut approximations = Vec::new();

    // Iterate until the convergence is achieved (difference is smaller than the tolerance).
    loop {
        // Compute the next iteration value.
        let next = g(current);
        // Add the next iteration value to the list of approximations.
        approximations.push(next);

        // If the difference is within the tolerance, return the current approximation.
        if (next - current).abs() < tolerance {
            return (next, approximations);
        }

        // Update the current guess.
        current = next;
    }
}

/// Performs Newton's method to approximate a root of the function f(x)
/// and records the approximations at each iteration.
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
/// A tuple containing an Option with the approximated root (or None if there is
/// a risk of dividing by zero) and a vector containing all the approximations.
fn newton(
    f: impl Fn(f64) -> f64,
    f_prime: impl Fn(f64) -> f64,
    start: f64,
    tolerance: f64,
) -> (Option<f64>, Vec<f64>) {
    // Define a vector for storing approximations.
    let mut approximations = Vec::new();
    // Set starting point to the initial guess.
    let mut current = start;

    // Iterate until the convergence is achieved (difference is smaller than the tolerance).
    loop {
        // Evaluate the function and derivative at the current approximation.
        let f_current = f(current);
        let fp_current = f_prime(current);

        // Check if the derivative is too small to avoid division by zero.
        if fp_current.abs() < std::f64::EPSILON {
            return (None, approximations);
        }

        // Compute the next approximation using Newton's formula: xₙₑₓₜ = x - f(x)/f'(x)
        let next = current - f_current / fp_current;
        // Add the calculated value to the list of approximations.
        approximations.push(next);

        // Check if the difference between successive approximations is within tolerance.
        if (next - current).abs() < tolerance {
            return (Some(next), approximations);
        }

        // Update the current approximation for the next iteration.
        current = next;
    }
}

/// Performs the secant method to approximate a root of the function f(x) = 0
/// using two initial guesses and records the approximations at each iteration.
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
/// A tuple containing an Option with the approximated root (or None if there is
/// a risk of dividing by zero) and a vector containing all the approximations.
fn secant(
    f: impl Fn(f64) -> f64,
    mut previous: f64,
    mut current: f64,
    tolerance: f64
) -> (Option<f64>, Vec<f64>) {
    // Define a vector for storing approximations.
    let mut approximations = Vec::new();

    // Iterate until the convergence is achieved (difference is smaller than the tolerance).
    loop {
        // Evaluate the function at the current approximation.
        let f_previous = f(previous);
        let f_current = f(current);

        // Check if the derivative is too small to avoid division by zero.
        if (f_current - f_previous).abs() < std::f64::EPSILON {
            return (None, approximations);
        }

        // Compute the next approximation using the secant formula:
        // x₂ = x₁ - f(x₁) * (x₁ - x₀) / (f(x₁) - f(x₀))
        let next = current - f_current * (current - previous) / (f_current - f_previous);
        // Add computed value to the list of approximations.
        approximations.push(next);

        // Check if the difference between successive approximations is within tolerance.
        if (next - current).abs() < tolerance {
            return (Some(next), approximations);
        }

        // Update the previous two approximations for the next iteration.
        previous = current;
        current = next;
    }
}

/// Computes error metrics for a sequence of approximations.
///
/// The errors are defined as follows:
///
/// - **Error 1:** |pₙ - pₙ₋₁| / |pₙ|
/// - **Error 2:** |pₙ - pₙ₋₁|
/// - **Error 3:** |f(pₙ)|
///
/// # Arguments
///
/// * `approximations` - A slice of approximations pₙ.
///
/// # Returns
///
/// A tuple of three vectors:
/// - The first vector contains Error 1 values.
/// - The second vector contains Error 2 values.
/// - The third vector contains Error 3 values.
fn compute_errors(approximations: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Store result for relative error metric: |pₙ - pₙ₋₁| / |pₙ|
    let mut error1 = Vec::new();
    // Store result for absolute error metric: |pₙ - pₙ₋₁|
    let mut error2 = Vec::new();
    // Store result for function error metric: |f(pₙ)|
    let mut error3 = Vec::new();

    // Iterate over the approximations from the second approximation since pₙ₋₁ is needed.
    for i in 1..approximations.len() {
        // Compute the absolute difference between current and previous approximations.
        let difference = (approximations[i] - approximations[i - 1]).abs();
        // Compute the relative error by dividing the difference
        // by the absolute value of the current approximation.
        let rel_error = difference / approximations[i].abs();

        // Append the computed values to each error metric.
        error1.push(rel_error);
        error2.push(difference);
        error3.push(f(approximations[i]).abs());
    }

    // Return a tuple containing the three error vectors.
    (error1, error2, error3)
}

/// Plots the error metrics on a graph using the plotters crate.
/// This function creates a PNG file with the error graphs.
///
/// # Arguments
///
/// * `error1` - A slice of relative error values (Error 1).
/// * `error2` - A slice of absolute error values (Error 2).
/// * `error3` - A slice of function error values (Error 3).
/// * `method` - A label for the method used in the filename and caption.
///
/// # Returns
///
/// A Result indicating success or failure.
fn plot_errors(
    error1: &[f64],
    error2: &[f64],
    error3: &[f64],
    method: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Compute the maximum error among all three error vectors to set the y-axis range.
    let max_error = error1
        .iter()
        .chain(error2.iter())
        .chain(error3.iter())
        .fold(0.0, |acc, &val| if val > acc { val } else { acc });

    // Create the "images" directory in the project root if it doesn't exist.
    let image_directory = "images";
    fs::create_dir_all(image_directory)?;

    // Store the filename in a variable so it lives long enough for use below.
    let filename = format!("{}/{}_errors.png", image_directory, method);

    // Initialize the drawing area for the chart with a resolution of 640x480.
    let root_area = BitMapBackend::new(&filename, (640, 480)).into_drawing_area();
    // Fill the entire drawing area with a white background.
    root_area.fill(&WHITE)?;

    // Create a chart builder on the drawing area with a title, margins, and label area sizes
    // The x-axis is set from 0 to the number of iterations and the y-axis is set from 0
    // to slightly above the maximum error.
    let mut chart = ChartBuilder::on(&root_area)
        .caption(format!("Error Metrics for {}", method), ("sans-serif", 20))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..(error1.len() as i32), 0f64..(max_error * 1.1))?;
    // Draw the mesh for the chart.
    chart.configure_mesh().draw()?;

    // Plot the relative error (error 1) as a red line.
    chart
        .draw_series(LineSeries::new(
            error1.iter().enumerate().map(|(i, &y)| (i as i32, y)),
            &RED,
        ))?
        .label("Relative Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot the absolute error (error 2) as a blue line.
    chart
        .draw_series(LineSeries::new(
            error2.iter().enumerate().map(|(i, &y)| (i as i32, y)),
            &BLUE,
        ))?
        .label("Absolute Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Plot the function error (error 3) as a green line.
    chart
        .draw_series(LineSeries::new(
            error3.iter().enumerate().map(|(i, &y)| (i as i32, y)),
            &GREEN,
        ))?
        .label("Function Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    // Configure and draw the legend with a border style.
    chart.configure_series_labels().border_style(&BLACK).draw()?;

    // Return an indicator that the chart was successfully created.
    Ok(())
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
fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    // Compute the approximate root using the bisection method.
    let (bisection_root, bisection_seq) = bisection(f, a, b, tolerance);
    // Print the approximate root for the bisection method to 7 decimal places.
    println!("Approximate root using the bisection method: {:.7}", bisection_root);
    // Compute the error metrics for the bisection method.
    let (bisection_error1, bisection_error2, bisection_error3) = compute_errors(&bisection_seq);
    // Plot the error metrics on a graph.
    plot_errors(&bisection_error1, &bisection_error2, &bisection_error3, "Bisection")?;

    // ---------------------------
    // Fixed-Point Iteration
    // ---------------------------
    // Compute the approximate root using the fixed-point iteration method.
    let (fpi_root, fpi_seq) = fixed_point(g, initial_guess, tolerance);
    // Print the approximate root for the fixed-point iteration method to 7 decimal places.
    println!("Approximate root using the fixed-point iteration method: {:.7}", fpi_root);
    // Compute the error metrics for the fixed-point iteration method.
    let (fpi_error1, fpi_error2, fpi_error3) = compute_errors(&fpi_seq);
    // Plot the error metrics on a graph.
    plot_errors(&fpi_error1, &fpi_error2, &fpi_error3, "Fixed-Point Iteration")?;

    // ---------------------------
    // Newton's Method
    // ---------------------------
    // Compute the approximate root using Newton's method.
    let (newton_result, newton_seq) = newton(f, f_prime, initial_guess, tolerance);
    // Print the result to 7 decimal places if available.
    if let Some(newton_root) = newton_result {
        println!("Approximate root using Newton's method: {:.7}", newton_root);
    } else {
        println!("Newton's method did not converge on an approximate root.");
    }
    // Compute the error metrics for Newton's method.
    let (newton_error1, newton_error2, newton_error3) = compute_errors(&newton_seq);
    // Plot the error metrics on a graph.
    plot_errors(&newton_error1, &newton_error2, &newton_error3, "Newton")?;

    // ---------------------------
    // Secant Method
    // ---------------------------
    // Compute the approximate root using the secant method.
    let (secant_result, secant_seq) = secant(f, a, b, tolerance);
    // Print the result to 7 decimal places if available.
    if let Some(secant_root) = secant_result {
        println!("Approximate root using the secant method: {:.7}", secant_root);
    } else {
        println!("Secant method did not converge on an approximate root.")
    }
    // Compute the error metrics for the secant method.
    let (secant_error1, secant_error2, secant_error3) = compute_errors(&secant_seq);
    // Plot the error metrics on a graph.
    plot_errors(&secant_error1, &secant_error2, &secant_error3, "Secant")?;

    // Exit program successfully.
    Ok(())
}
