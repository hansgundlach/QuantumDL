import importlib
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from copy import deepcopy
from decimal import Decimal, getcontext, DecimalException
from scipy.optimize import curve_fit
from matplotlib.font_manager import FontProperties
import quantum_dl_lib


# enhanced binary search intersection robust version for large numbers
# default tolerance is 1e-5
def binary_search_intersection(
    func1, func2, low, high, tolerance=1e-9, max_iterations=100000
) -> float:
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
    low = Decimal(str(low))
    high = Decimal(str(high))
    tol = Decimal(str(tolerance))

    def safe_f(x_dec: Decimal) -> float:
        """
        Evaluate f(x)=func1(x)-func2(x) in a way that avoids overflow.

        If one (or both) function evaluations results in a value that overflows (or is infinite),
        we assign float('inf') to that value. In the case that both are infinite (e.g. for enormous x)
        we instead compare their logarithms.

        Returns:
            A float representing the difference, or the difference of logarithms if needed.
        """
        x = float(x_dec)
        try:
            v1 = func1(x)
        except OverflowError:
            v1 = float("inf")
        try:
            v2 = func2(x)
        except OverflowError:
            v2 = float("inf")

        # If either function evaluates to infinity (or a value that compares as such), we replace it.
        if math.isinf(v1) or math.isinf(v2):
            # If both are infinite and positive, try to compare their logarithms.
            if v1 > 0 and v2 > 0 and math.isinf(v1) and math.isinf(v2):
                try:
                    log_v1 = math.log(func1(x))
                except (OverflowError, ValueError):
                    log_v1 = float("inf")
                try:
                    log_v2 = math.log(func2(x))
                except (OverflowError, ValueError):
                    log_v2 = float("inf")
                return log_v1 - log_v2
            # Otherwise, if only one is infinite, the difference will have the sign of the finite number.
            # (If v1 is infinite and v2 is not, f(x) is positive; if v2 is infinite, f(x) is negative.)
            if math.isinf(v1) and not math.isinf(v2):
                return float("inf")
            if math.isinf(v2) and not math.isinf(v1):
                return -float("inf")
        return v1 - v2

    # Evaluate at the endpoints.
    try:
        f_low = safe_f(low)
        f_high = safe_f(high)
    except (OverflowError, ValueError, DecimalException) as e:
        print("Error evaluating function at endpoints:", e)
        return None

    # Check that f(low) and f(high) bracket a sign change.
    if f_low * f_high > 0:
        # print("No sign change in f(x) over the interval; intersection not guaranteed.")
        return None

    for i in range(max_iterations):
        mid = (low + high) / 2
        f_mid = safe_f(mid)

        # If we are close enough, return.
        if abs(f_mid) < float(tol):
            return float(mid)

        # Decide which side of the interval contains the sign change.
        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

        # If the interval has shrunk sufficiently, exit.
        if abs(high - low) < tol:
            return float(mid)

    # print(f"No intersection found within {max_iterations} iterations")
    return None


# #constants in different format
fidelity_improvement_rate = 0.28
gate_speed_improvement_rate = 0.14
classical_speed_init = 1 / (1.5 * 1e9)  # Nividia H100 GPU clock speed
superconducting_gate_speed_init = 1e-6  # (1/(1.5*1e9))*(10**3.78) # seconds #overhead taken from quantum economic advantage calculator
initial_error = 10 ** (-2.5)  # initial error from error graphs
classical_speed_improvement_rate = 0.3  # moore's law improvement
number_of_processors = 1e8  # processor overehead done from calculations for GPUs
# time_upper_limt = 3.14*1e7 # 1-year of seconds
connectivity_penalty_exponent = 0.0  # connectivity penalty for physical to logical qubit ratio in this range. default no asymptotic connectivity penality
time_upper_limit = (3.14 * 1e7) / 12  # 1 month computation time
scode_init_speed_overhead = 1e2  # slowdown overhead from Choi, Neil, and Moses
alg_overhead_qubit = 1e0  # algorithm overhead in logical qubits
alg_overhead_qspeed = 1e0  # algorithm speed overhead based on constants this is exclusivly for quantum algorithm
classical_alg_overhead = 1e0
MAX_PROBLEM_SIZE = 1e51
MIN_YEAR = 2025
MAX_YEAR = 2050

# aggressive projection
# constants in different format
# fidelity_improvement_rate = .28
# gate_speed_improvement_rate = .14
# classical_speed_init = 1/(5*1e9) # seconds
# superconducting_gate_speed_init = (1/(5*1e9))*(10**3.78) # seconds #overhead taken from quantum economic advantage calculator
# initial_error = 10**(-2.5)
# classical_speed_improvement_rate = 0.3
# number_of_processors = 1e5
# # time_upper_limt = 3.14*1e7 # 1-year of seconds
# connectivity_penalty_exponent = 0.0 #connectivity penalty for physical to logical qubit ratio in this range.
# time_upper_limit = (3.14*1e7)/12 # 1 week computation time
# surface_code_overhead = 1e2 # number from Choi, Neil, and Moses
# MAX_PROBLEM_SIZE = 1e50
# MIN_YEAR = 2025
# MAX_YEAR = 2050

# default quantum and classical runtime if not specified
classical_runtime = "n**3"
quantum_runtime = "n"

IBM_ROADMAP = {
    2020: 27,
    2022: 127,
    2024: 133,
}
GOOGLE_ROADMAP = {
    2019: 53,
    2024: 105,
}
PERCENTILE_95_ROADMAP = {
    2025: 1662,
    2030: 15660,
}
SOTA_ROADMAP = {
    2025: 1959,
    2030: 24352,
}
percentile_90_roadmap = {2025: 305.13, 2030: 1315}

default_roadmap = percentile_90_roadmap


# gives the physical to logical overhead based on the surface code formula
def surface_code_formula(pP: float) -> float:
    pL = 1e-18
    pth = 1e-2
    numerator = 4 * math.log(math.sqrt(10 * pP / pL))
    denominator = math.log(pth / pP)
    fraction = numerator / denominator
    f_QEC = (fraction + 1) ** -2
    return f_QEC**-1


def problem_size_qubit_feasible(
    year: int,
    roadmap: dict = default_roadmap,
    alg_overhead: float = 1,
    q_prob_size="log",
) -> float:

    # fit exponential to roadmap
    years = np.array(list(roadmap.keys()))
    qubits = np.array(list(roadmap.values()))
    min_year = min(years)
    # initial gues
    p0 = [min(qubits), 0.5, 0]

    # fit an exponential curve to the data
    def exp_func(x, a, b, c):
        return a * np.exp(b * (x - min_year)) + c

    try:
        popt, _ = curve_fit(
            exp_func, years, qubits, p0=p0, bounds=([0, -2, -1000], [10000, 2, 1000])
        )
    except RuntimeError:
        # print(f"Fitting failed for {label}")
        return None
    surf_overhead = surface_code_formula(
        initial_error * (1 - fidelity_improvement_rate) ** (year - 2025)
    )
    if q_prob_size == "log":
        qubit_number = min(
            exp_func(year, *popt) / (surf_overhead * alg_overhead_qubit),
            np.log2(MAX_PROBLEM_SIZE),
        )
        return 2 ** (qubit_number)
    else:
        return min(
            exp_func(year, *popt) / (surf_overhead * alg_overhead_qubit),
            MAX_PROBLEM_SIZE,
        )


# quantum speed per operation function
def quantum_seconds_per_operation(year):
    gate_speed = superconducting_gate_speed_init * (
        1 - gate_speed_improvement_rate
    ) ** (year - 2025)
    # with error correction
    fidelity_year = initial_error * (1 - fidelity_improvement_rate) ** (year - 2025)
    proportional_change = (surface_code_formula(fidelity_year)) / surface_code_formula(
        initial_error
    )
    return gate_speed * scode_init_speed_overhead * proportional_change ** (1.5)


# seconds per effective operation ie here we factor in parallelism
def classical_seconds_per_operation(year):
    # just dividing by number of processors here for now to simplify things
    return (
        1e-9
        * (1 - classical_speed_improvement_rate) ** (year - 2025)
        / number_of_processors
    )


# maximum problem size that can be solved on a quantum computer in a given amount of time at a given year
def find_largest_problem_size(
    runtime_string,
    year: int,
    quantum=True,
    qadv_only=False,
    roadmap: dict = default_roadmap,
    stagnation_year=2200,
    time_upper_limit=time_upper_limit,
    q_prob_size="log",
) -> float:
    # Constants
    try:
        # Convert string expression to lambda function
        n = sp.Symbol("n")
        expr = sp.sympify(runtime_string)
        expr = expr * n**connectivity_penalty_exponent * alg_overhead_qspeed
        # Apply connectivity penalty to the runtime
        runtime_func = sp.lambdify(n, expr)

        if quantum:
            quantum_total = lambda size: quantum_seconds_per_operation(
                year
            ) * runtime_func(size)

            def quantum_limit(x):
                return time_upper_limit  # number of seconds in a year
                # return q_ops_second_dollar

            qadv = binary_search_intersection(
                quantum_total, quantum_limit, low=2.5, high=1e50
            )
            if qadv is None:
                # print(f"No intersection found between quantum total and limit for year {year}")
                qadv = float("inf")
            # except Exception as e:
            #     print(f"Error finding intersection for year {year}: {e}")
            #     return float("inf")

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

            def classical_cost(x):
                if year < stagnation_year:
                    return classical_seconds_per_operation(year) * runtime_func(x)
                else:
                    return classical_seconds_per_operation(
                        stagnation_year
                    ) * runtime_func(x)

            def classical_limit(x):
                return time_upper_limit
                # return ops_second_dollar

            return binary_search_intersection(
                classical_cost, classical_limit, low=2.5, high=1e50
            )

    except Exception as e:
        print(f"Error in function evaluation: {e}")
        return None


def quantum_advantage_size_by_year(
    year, classical_runtime_string: str, quantum_runtime_string: str
) -> float:
    n = sp.symbols("n")
    class_expr = sp.sympify(classical_runtime_string)
    class_expr = (
        class_expr * classical_seconds_per_operation(year) * classical_alg_overhead
    )
    classical_runtime_func = sp.lambdify(n, class_expr)

    quant_expr = sp.sympify(quantum_runtime_string)
    quant_expr = quant_expr * n**connectivity_penalty_exponent * alg_overhead_qspeed
    quant_expr = quant_expr * quantum_seconds_per_operation(year)
    quantum_runtime_func = sp.lambdify(n, quant_expr)

    # Find intersection using the original string expressions
    return binary_search_intersection(
        classical_runtime_func, quantum_runtime_func, 2.5, 1e50
    )


# this is the intersection of quantum advantage size by year and quantum problem size qubit feasible by year
def generalized_qea(
    classical_runtime_string: str, quantum_runtime_string: str
) -> float:
    return binary_search_intersection(
        lambda x: quantum_advantage_size_by_year(
            x, classical_runtime_string, quantum_runtime_string
        ),
        lambda x: find_largest_problem_size(quantum_runtime_string, x, quantum=True),
        MIN_YEAR,
        MAX_YEAR,
    )
