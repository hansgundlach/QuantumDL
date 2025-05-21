import math
from decimal import Decimal, getcontext, InvalidOperation, DecimalException
import numpy as np
from scipy.optimize import curve_fit


def binary_search_intersection(
    func1, func2, low, high, tolerance=1e-9, max_iterations=100000
):
    """
    Find the intersection of two functions f1(x)=func1(x) and f2(x)=func2(x) (i.e. where f1(x)==f2(x))
    using a robust binary search approach that can handle enormous numbers.

    This version avoids evaluating huge numbers directly by using a "safe" evaluation routine.
    It assumes that on the search interval the functions are positive (so logarithms make sense)
    and that the difference f(x)=func1(x)-func2(x) is monotonic.

    Args:
        func1, func2: The two functions.
        low, high: The endpoints of the search interval.
        tolerance: The acceptable error for f(x) (or its logarithmic version).
        max_iterations: Maximum iterations to try.

    Returns:
        The x-value where the functions intersect, or None if no intersection was found.
    """
    # Use high precision for interval arithmetic.
    getcontext().prec = 100

    try:
        low = Decimal(str(low))
        high = Decimal(str(high))
        tol = Decimal(str(tolerance))
    except (InvalidOperation, TypeError) as e:
        print(f"Error in binary_search_intersection parameters: {e}")
        return None

    def safe_f(x_dec: Decimal) -> float:
        """
        Evaluate f(x)=func1(x)-func2(x) in a way that avoids overflow.
        Returns None if evaluation is not possible or values are not comparable.
        """
        try:
            x = float(x_dec)

            # Try to get values, handling potential errors
            try:
                v1 = func1(x)
            except (OverflowError, ValueError, TypeError, ZeroDivisionError) as e:
                v1 = float("inf")

            try:
                v2 = func2(x)
            except (OverflowError, ValueError, TypeError, ZeroDivisionError) as e:
                v2 = float("inf")

            # Check if either value is None and handle it
            if v1 is None or v2 is None:
                return None  # Return None to indicate no valid comparison

            # If either function evaluates to infinity, handle it carefully
            if math.isinf(v1) or math.isinf(v2):
                # If both are infinite, try logarithmic comparison if possible
                if v1 > 0 and v2 > 0 and math.isinf(v1) and math.isinf(v2):
                    try:
                        log_v1 = math.log(func1(x))
                    except (OverflowError, ValueError, TypeError, ZeroDivisionError):
                        log_v1 = float("inf")
                    try:
                        log_v2 = math.log(func2(x))
                    except (OverflowError, ValueError, TypeError, ZeroDivisionError):
                        log_v2 = float("inf")
                    return log_v1 - log_v2

                # Handle case where only one is infinite
                if math.isinf(v1) and not math.isinf(v2):
                    return float("inf")
                if math.isinf(v2) and not math.isinf(v1):
                    return -float("inf")

            return v1 - v2
        except Exception as e:
            print(f"Unexpected error in safe_f: {e}")
            return None

    # Evaluate at the endpoints.
    try:
        f_low = safe_f(low)
        f_high = safe_f(high)
    except Exception as e:
        print(f"Error evaluating function at endpoints: {e}")
        return None

    # Check if either value is None
    if f_low is None or f_high is None:
        print("Function evaluation returned None at endpoints")
        return None

    # Check that f(low) and f(high) bracket a sign change.
    try:
        product = f_low * f_high
        if product > 0:  # No sign change
            return None
    except (TypeError, ValueError) as e:
        print(f"Error checking sign change: {e}")
        return None

    # Binary search
    try:
        for i in range(max_iterations):
            mid = (low + high) / 2
            f_mid = safe_f(mid)

            # Handle None result from safe_f
            if f_mid is None:
                print(f"Function evaluation returned None at x={mid}")
                return None

            # If we are close enough, return.
            if abs(f_mid) < float(tol):
                return float(mid)

            # Decide which side of the interval contains the sign change.
            try:
                if f_low * f_mid < 0:
                    high = mid
                    f_high = f_mid
                else:
                    low = mid
                    f_low = f_mid
            except (TypeError, ValueError) as e:
                print(f"Error updating interval: {e}")
                return None

            # If the interval has shrunk sufficiently, exit.
            if abs(high - low) < tol:
                return float(mid)
    except Exception as e:
        print(f"Error in binary search: {e}")
        return None

    return None


def surface_code_formula(pP: float) -> float:
    """
    Calculate the physical to logical overhead based on the surface code formula

    Args:
        pP: Physical error rate

    Returns:
        Surface code overhead factor
    """
    if pP is None or pP <= 0:
        print("Invalid physical error rate for surface code calculation")
        return float("inf")  # Return infinity for invalid inputs

    try:
        pL = 1e-18
        pth = 1e-2
        numerator = 4 * math.log(math.sqrt(10 * pP / pL))
        denominator = math.log(pth / pP)
        fraction = numerator / denominator
        f_QEC = (fraction + 1) ** -2
        return f_QEC**-1
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        print(f"Error in surface code calculation: {e}")
        return float("inf")  # Return infinity for calculation errors


def problem_size_qubit_feasible(
    roadmap: dict,
    year: int,
    alg_overhead: float = 1,
    q_prob_size="log",
    initial_error=10 ** (-2.5),
    fidelity_improvement_rate=0.28,
    MAX_PROBLEM_SIZE=1e50,
) -> float:
    """
    Calculate the maximum problem size that is feasible with the given qubit roadmap

    Args:
        roadmap: Dictionary mapping years to number of qubits
        year: Year to calculate for
        alg_overhead: Algorithm overhead in logical qubits
        q_prob_size: Problem size scaling ("log" or linear)
        initial_error: Initial physical error rate
        fidelity_improvement_rate: Annual rate of improvement in fidelity
        MAX_PROBLEM_SIZE: Maximum allowable problem size

    Returns:
        Maximum feasible problem size or None if calculation fails
    """
    if not roadmap or year is None:
        print("Invalid roadmap or year")
        return None

    try:
        # fit exponential to roadmap
        years = np.array(list(roadmap.keys()))
        qubits = np.array(list(roadmap.values()))
        min_year = min(years)

        # initial guess
        p0 = [min(qubits), 0.5, 0]

        # fit an exponential curve to the data
        def exp_func(x, a, b, c):
            return a * np.exp(b * (x - min_year)) + c

        try:
            popt, _ = curve_fit(
                exp_func,
                years,
                qubits,
                p0=p0,
                bounds=([0, -2, -1000], [10000, 2, 1000]),
            )
        except (RuntimeError, ValueError) as e:
            print(f"Fitting failed: {e}")
            return None

        # Calculate surface code overhead
        error_at_year = initial_error * (1 - fidelity_improvement_rate) ** (year - 2025)
        surf_overhead = surface_code_formula(error_at_year)

        if math.isinf(surf_overhead) or surf_overhead <= 0:
            print(f"Invalid surface code overhead for year {year}")
            return None

        # Calculate qubit number based on problem size scaling
        if q_prob_size == "log":
            qubit_number = min(
                exp_func(year, *popt) / (surf_overhead * alg_overhead),
                np.log2(MAX_PROBLEM_SIZE),
            )
            return 2 ** (qubit_number)
        else:
            return min(
                exp_func(year, *popt) / (surf_overhead * alg_overhead),
                MAX_PROBLEM_SIZE,
            )
    except Exception as e:
        print(f"Error in problem_size_qubit_feasible: {e}")
        return None


def quantum_advantage_size_by_year(
    year,
    classical_runtime_string: str,
    quantum_runtime_string: str,
    classical_seconds_per_operation,
    quantum_seconds_per_operation,
    connectivity_penalty_exponent=0.0,
    classical_alg_overhead=1.0,
    alg_overhead_qspeed=1.0,
) -> float:
    """
    Calculate the problem size where quantum computing gives an advantage in the specified year

    Args:
        year: Year to calculate for
        classical_runtime_string: Expression for classical runtime complexity
        quantum_runtime_string: Expression for quantum runtime complexity
        classical_seconds_per_operation: Function that returns classical seconds per operation for a given year
        quantum_seconds_per_operation: Function that returns quantum seconds per operation for a given year
        connectivity_penalty_exponent: Exponent for connectivity penalty
        classical_alg_overhead: Classical algorithm overhead factor
        alg_overhead_qspeed: Quantum algorithm speed overhead factor

    Returns:
        Problem size where quantum computing gives an advantage, or None if there is no advantage
    """
    try:
        import sympy as sp

        n = sp.Symbol("n")

        # Parse expressions
        try:
            class_expr = sp.sympify(classical_runtime_string)
            class_expr = (
                class_expr
                * classical_seconds_per_operation(year)
                * classical_alg_overhead
            )
            classical_runtime_func = sp.lambdify(n, class_expr)
        except (ValueError, TypeError, SyntaxError) as e:
            print(f"Error parsing classical runtime expression: {e}")
            return None

        try:
            quant_expr = sp.sympify(quantum_runtime_string)
            quant_expr = (
                quant_expr * n**connectivity_penalty_exponent * alg_overhead_qspeed
            )
            quant_expr = quant_expr * quantum_seconds_per_operation(year)
            quantum_runtime_func = sp.lambdify(n, quant_expr)
        except (ValueError, TypeError, SyntaxError) as e:
            print(f"Error parsing quantum runtime expression: {e}")
            return None

        # Find intersection using binary search
        return binary_search_intersection(
            classical_runtime_func, quantum_runtime_func, 2.5, 1e50
        )
    except Exception as e:
        print(f"Error in quantum_advantage_size_by_year: {e}")
        return None


def find_largest_problem_size(
    runtime_string,
    year: int,
    quantum=True,
    qadv_only=False,
    roadmap=None,
    stagnation_year=2200,
    time_upper_limit=None,
    q_prob_size="log",
    quantum_seconds_per_operation=None,
    classical_seconds_per_operation=None,
    connectivity_penalty_exponent=0.0,
    alg_overhead_qspeed=1.0,
    MAX_PROBLEM_SIZE=1e50,
) -> float:
    """
    Find the largest problem size that can be solved given constraints

    Args:
        runtime_string: Expression for runtime complexity
        year: Year to calculate for
        quantum: Whether to calculate for quantum computing
        qadv_only: Whether to only consider quantum advantage
        roadmap: Qubit roadmap dictionary
        stagnation_year: Year after which improvements stagnate
        time_upper_limit: Maximum computation time allowed
        q_prob_size: Problem size scaling ("log" or linear)
        quantum_seconds_per_operation: Function to calculate quantum seconds per operation
        classical_seconds_per_operation: Function to calculate classical seconds per operation
        connectivity_penalty_exponent: Exponent for connectivity penalty
        alg_overhead_qspeed: Quantum algorithm speed overhead factor
        MAX_PROBLEM_SIZE: Maximum allowable problem size

    Returns:
        Largest problem size or None if calculation fails
    """
    if not time_upper_limit or not roadmap:
        print("Missing required parameters")
        return None

    try:
        # Convert string expression to lambda function
        import sympy as sp

        n = sp.Symbol("n")

        try:
            expr = sp.sympify(runtime_string)
            expr = expr * n**connectivity_penalty_exponent * alg_overhead_qspeed
            # Apply connectivity penalty to the runtime
            runtime_func = sp.lambdify(n, expr)
        except (ValueError, TypeError, SyntaxError) as e:
            print(f"Error parsing runtime expression: {e}")
            return None

        if quantum:
            if quantum_seconds_per_operation is None:
                print("Missing quantum_seconds_per_operation function")
                return None

            quantum_total = lambda size: quantum_seconds_per_operation(
                year
            ) * runtime_func(size)

            def quantum_limit(x):
                return time_upper_limit  # number of seconds in a year

            qadv = binary_search_intersection(
                quantum_total, quantum_limit, low=2.5, high=1e50
            )

            if qadv is None:
                qadv = float("inf")

            size_feasible = problem_size_qubit_feasible(
                roadmap=roadmap, year=year, q_prob_size=q_prob_size
            )

            # Handle the case where size_feasible is None
            if size_feasible is None:
                print(f"No feasible size found for year {year}. Returning infinity.")
                return float("inf")  # Return infinity if no feasible size is found

            if not qadv_only:
                return min(qadv, size_feasible)
            else:
                return qadv
        else:
            if classical_seconds_per_operation is None:
                print("Missing classical_seconds_per_operation function")
                return None

            def classical_cost(x):
                if year < stagnation_year:
                    return classical_seconds_per_operation(year) * runtime_func(x)
                else:
                    return classical_seconds_per_operation(
                        stagnation_year
                    ) * runtime_func(x)

            def classical_limit(x):
                return time_upper_limit

            result = binary_search_intersection(
                classical_cost, classical_limit, low=2.5, high=1e50
            )
            return result
    except Exception as e:
        print(f"Error in find_largest_problem_size: {e}")
        return None


def generalized_qea(
    classical_runtime_string: str,
    quantum_runtime_string: str,
    MIN_YEAR=2025,
    MAX_YEAR=2050,
    quantum_advantage_size_by_year_func=None,
    find_largest_problem_size_func=None,
) -> float:
    """
    Find the year when quantum economic advantage is achieved

    Args:
        classical_runtime_string: Expression for classical runtime complexity
        quantum_runtime_string: Expression for quantum runtime complexity
        MIN_YEAR: Minimum year to consider
        MAX_YEAR: Maximum year to consider
        quantum_advantage_size_by_year_func: Function to calculate quantum advantage size by year
        find_largest_problem_size_func: Function to find largest problem size

    Returns:
        Year of quantum economic advantage or None if not found
    """
    if not quantum_advantage_size_by_year_func or not find_largest_problem_size_func:
        print("Missing required functions")
        return None

    try:
        # Function to find the intersection
        return binary_search_intersection(
            lambda x: quantum_advantage_size_by_year_func(
                x, classical_runtime_string, quantum_runtime_string
            ),
            lambda x: find_largest_problem_size_func(
                quantum_runtime_string, x, quantum=True
            ),
            MIN_YEAR,
            MAX_YEAR,
        )
    except Exception as e:
        print(f"Error in generalized_qea: {e}")
        return None
