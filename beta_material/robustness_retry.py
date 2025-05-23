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
    try:
        # Use high precision for interval arithmetic.
        getcontext().prec = 100
        low = Decimal(str(low))
        high = Decimal(str(high))
        tol = Decimal(str(tolerance))

        def safe_f(x_dec: Decimal) -> float:
            """
            Evaluate f(x)=func1(x)-func2(x) in a way that avoids overflow.
            """
            x = float(x_dec)
            try:
                v1 = func1(x)
            except (OverflowError, ValueError, TypeError, ZeroDivisionError) as e:
                # print(f"Error evaluating func1({x}): {e}")
                v1 = float("inf")
            try:
                v2 = func2(x)
            except (OverflowError, ValueError, TypeError, ZeroDivisionError) as e:
                # print(f"Error evaluating func2({x}): {e}")
                v2 = float("inf")

            # Check if either value is None and handle it
            if v1 is None or v2 is None:
                return None  # Return None to indicate no valid comparison

            # If either function evaluates to infinity (or a value that compares as such), we replace it.
            if math.isinf(v1) or math.isinf(v2):
                # If both are infinite and positive, try to compare their logarithms.
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
        except (OverflowError, ValueError, DecimalException, TypeError, ZeroDivisionError) as e:
            # print("Error evaluating function at endpoints:", e)
            return None

        # Check if either value is None
        if f_low is None or f_high is None:
            # print("Function evaluation returned None at endpoints")
            return None

        # Check that f(low) and f(high) bracket a sign change.
        if f_low * f_high > 0:
            # print("No sign change in f(x) over the interval; intersection not guaranteed.")
            return None

        for i in range(max_iterations):
            mid = (low + high) / 2
            try:
                f_mid = safe_f(mid)
                
                # Handle case where f_mid is None
                if f_mid is None:
                    # print(f"Function evaluation returned None at x={mid}")
                    return None
                
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
                
            except (OverflowError, ValueError, DecimalException, TypeError, ZeroDivisionError) as e:
                # print(f"Error during binary search at iteration {i}, x={mid}: {e}")
                return None

        # print(f"No intersection found within {max_iterations} iterations")
        return None
        
    except Exception as e:
        # print(f"Unexpected error in binary_search_intersection: {e}")
        return None


# #constants in different format
fidelity_improvement_rate = 0.28
gate_speed_improvement_rate = 0.14
classical_speed_init = 1 / (1.5 * 1e9)  #
superconducting_gate_speed_init = 1e-6  # (1/(1.5*1e9))*(10**3.78) # seconds #overhead taken from quantum economic advantage calculator
initial_error = 10 ** (-2.5)
classical_speed_improvement_rate = 0.3  # moore's law improvement
number_of_processors = 1e8  # processor overehead done from calculations for GPUs
# time_upper_limt = 3.14*1e7 # 1-year of seconds
connectivity_penalty_exponent = 0.0  # connectivity penalty for physical to logical qubit ratio in this range. default no asymptotic connectivity penality
time_upper_limit = 4 * (3.14 * 1e7) / 52  # 1 month computation time
scode_init_speed_overhead = 1e2  # slowdown overhead from Choi, Neil, and Moses
alg_overhead_qubit = 1e1  # algorithm overhead in logical qubits
alg_overhead_qspeed = 1e0  # algorithm speed overhead based on constants this is exclusivly for quantum algorithm
classical_alg_overhead = 1e0
MAX_PROBLEM_SIZE = 1e50
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
    try:
        pL = 1e-18
        pth = 1e-2
        numerator = 4 * math.log(math.sqrt(10 * pP / pL))
        denominator = math.log(pth / pP)
        fraction = numerator / denominator
        f_QEC = (fraction + 1) ** -2
        return f_QEC**-1
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        # print(f"Error in surface_code_formula with pP={pP}: {e}")
        return float('inf')  # Return infinity for invalid inputs


#


def problem_size_qubit_feasible(
    roadmap: dict, year: int, alg_overhead: float = 1, q_prob_size="log"
) -> float:
    try:
        # Check if roadmap is valid
        if not roadmap or not isinstance(roadmap, dict):
            # print("Invalid roadmap provided")
            return None

        # fit exponential to roadmap
        years = np.array(list(roadmap.keys()))
        qubits = np.array(list(roadmap.values()))
        
        if len(years) < 2 or len(qubits) < 2:
            # print("Roadmap needs at least two data points")
            return None
            
        min_year = min(years)
        # initial guess
        p0 = [min(qubits), 0.5, 0]

        # fit an exponential curve to the data
        def exp_func(x, a, b, c):
            return a * np.exp(b * (x - min_year)) + c

        try:
            popt, _ = curve_fit(
                exp_func, years, qubits, p0=p0, bounds=([0, -2, -1000], [10000, 2, 1000])
            )
        except (RuntimeError, ValueError, TypeError) as e:
            # print(f"Fitting failed: {e}")
            return None
            
        try:
            surf_overhead = surface_code_formula(
                initial_error * (1-fidelity_improvement_rate) ** (year - 2025)
            )
            
            if surf_overhead <= 0 or math.isinf(surf_overhead):
                # print(f"Invalid surface code overhead: {surf_overhead}")
                return None
                
            if q_prob_size == "log":
                qubit_number = min(
                    exp_func(year, *popt) / (surf_overhead * alg_overhead_qubit),
                    np.log2(MAX_PROBLEM_SIZE),
                )
                if qubit_number <= 0:
                    # print(f"Invalid qubit number: {qubit_number}")
                    return None
                return 2 ** (qubit_number)
            else:
                return min(
                    exp_func(year, *popt) / (surf_overhead * alg_overhead_qubit),
                    MAX_PROBLEM_SIZE,
                )
        except (ValueError, OverflowError, ZeroDivisionError) as e:
            # print(f"Error calculating problem size: {e}")
            return None
            
    except Exception as e:
        # print(f"Unexpected error in problem_size_qubit_feasible: {e}")
        return None


# quantum speed per operation function
def quantum_seconds_per_operation(year):
    try:
        gate_speed = superconducting_gate_speed_init * (
            1 - gate_speed_improvement_rate
        ) ** (year - 2025)
        
        # with error correction
        fidelity_year = initial_error * (1-fidelity_improvement_rate) ** (year - 2025)
        
        if fidelity_year <= 0:
            # print(f"Invalid fidelity for year {year}: {fidelity_year}")
            return float('inf')
            
        current_overhead = surface_code_formula(fidelity_year)
        initial_overhead = surface_code_formula(initial_error)
        
        if math.isinf(current_overhead) or math.isinf(initial_overhead) or initial_overhead == 0:
            # print(f"Invalid surface code overhead calculation for year {year}")
            return float('inf')
            
        proportional_change = current_overhead / initial_overhead
        
        return gate_speed * scode_init_speed_overhead * proportional_change ** (1.5)
        
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        # print(f"Error in quantum_seconds_per_operation for year {year}: {e}")
        return float('inf')


# seconds per effective operation ie here we factor in parallelism
def classical_seconds_per_operation(year):
    try:
        # just dividing by number of processors here for now to simplify things
        return (
            classical_speed_init
            * (1 - classical_speed_improvement_rate) ** (year - 2025)
            / number_of_processors
        )
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        # print(f"Error in classical_seconds_per_operation for year {year}: {e}")
        return float('inf')


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







# # To add a new cell, type '# %%'
# # To add a new markdown cell, type '# %% [markdown]'
# # %%
# import importlib
# import sympy as sp
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# import pandas as pd
# from copy import deepcopy
# from decimal import Decimal, getcontext, DecimalException
# import math
# from scipy.optimize import curve_fit
# import sympy as sp
# from matplotlib.font_manager import FontProperties
# import quantum_dl_lib

# # importlib.reload(quantum_dl_lib)
# # from quantum_dl_lib import *


# #


# # %%
# # enhanced binary search intersection robust version for large numbers
# # default tolerance is 1e-5
# def binary_search_intersection(
#     func1, func2, low, high, tolerance=1e-9, max_iterations=100000
# ) -> float:
#     """
#     Find the intersection of two functions f1(x)=func1(x) and f2(x)=func2(x) (i.e. where f1(x)==f2(x))
#     using a robust binary search approach that can handle enormous numbers.

#     This version avoids evaluating huge numbers directly by using a "safe" evaluation routine.
#     It assumes that on the search interval the functions are positive (so logarithms make sense)
#     and that the difference f(x)=func1(x)-func2(x) is monotonic.

#     Args:
#         func1, func2: The two functions.
#         low, high: The endpoints of the search interval.
#         tolerance: The acceptable error for f(x) (or its logarithmic version).
#         max_iterations: Maximum iterations to try.

#     Returns:
#         The x-value where the functions intersect, or None if no intersection was found.
#     """
#     # Use high precision for interval arithmetic.
#     getcontext().prec = 100
#     low = Decimal(str(low))
#     high = Decimal(str(high))
#     tol = Decimal(str(tolerance))

#     def safe_f(x_dec: Decimal) -> float:
#         """
#         Evaluate f(x)=func1(x)-func2(x) in a way that avoids overflow.
#         """
#         x = float(x_dec)
#         try:
#             v1 = func1(x)
#         except OverflowError:
#             v1 = float("inf")
#         try:
#             v2 = func2(x)
#         except OverflowError:
#             v2 = float("inf")

#         # Check if either value is None and handle it
#         if v1 is None or v2 is None:
#             return None  # Return None to indicate no valid comparison

#         # If either function evaluates to infinity (or a value that compares as such), we replace it.
#         if math.isinf(v1) or math.isinf(v2):
#             # If both are infinite and positive, try to compare their logarithms.
#             if v1 > 0 and v2 > 0 and math.isinf(v1) and math.isinf(v2):
#                 try:
#                     log_v1 = math.log(func1(x))
#                 except (OverflowError, ValueError):
#                     log_v1 = float("inf")
#                 try:
#                     log_v2 = math.log(func2(x))
#                 except (OverflowError, ValueError):
#                     log_v2 = float("inf")
#                 return log_v1 - log_v2
#             # Otherwise, if only one is infinite, the difference will have the sign of the finite number.
#             # (If v1 is infinite and v2 is not, f(x) is positive; if v2 is infinite, f(x) is negative.)
#             if math.isinf(v1) and not math.isinf(v2):
#                 return float("inf")
#             if math.isinf(v2) and not math.isinf(v1):
#                 return -float("inf")
#         return v1 - v2

#     # Evaluate at the endpoints.
#     try:
#         f_low = safe_f(low)
#         f_high = safe_f(high)
#     except (OverflowError, ValueError, DecimalException) as e:
#         print("Error evaluating function at endpoints:", e)
#         return None

#     # Check if either value is None
#     if f_low is None or f_high is None:
#         return None

#     # Check that f(low) and f(high) bracket a sign change.
#     if f_low * f_high > 0:
#         # print("No sign change in f(x) over the interval; intersection not guaranteed.")
#         return None

#     for i in range(max_iterations):
#         mid = (low + high) / 2
#         f_mid = safe_f(mid)

#         # If we are close enough, return.
#         if abs(f_mid) < float(tol):
#             return float(mid)

#         # Decide which side of the interval contains the sign change.
#         if f_low * f_mid < 0:
#             high = mid
#             f_high = f_mid
#         else:
#             low = mid
#             f_low = f_mid

#         # If the interval has shrunk sufficiently, exit.
#         if abs(high - low) < tol:
#             return float(mid)

#     # print(f"No intersection found within {max_iterations} iterations")
#     return None


# # #constants in different format
# fidelity_improvement_rate = 0.28
# gate_speed_improvement_rate = 0.14
# classical_speed_init = 1 / (1.5 * 1e9)  #
# superconducting_gate_speed_init = 1e-6  # (1/(1.5*1e9))*(10**3.78) # seconds #overhead taken from quantum economic advantage calculator
# initial_error = 10 ** (-2.5)
# classical_speed_improvement_rate = 0.3  # moore's law improvement
# number_of_processors = 1e8  # processor overehead done from calculations for GPUs
# # time_upper_limt = 3.14*1e7 # 1-year of seconds
# connectivity_penalty_exponent = 0.0  # connectivity penalty for physical to logical qubit ratio in this range. default no asymptotic connectivity penality
# time_upper_limit = 4 * (3.14 * 1e7) / 52  # 1 month computation time
# scode_init_speed_overhead = 1e2  # slowdown overhead from Choi, Neil, and Moses
# alg_overhead_qubit = 1e1  # algorithm overhead in logical qubits
# alg_overhead_qspeed = 1e0  # algorithm speed overhead based on constants this is exclusivly for quantum algorithm
# classical_alg_overhead = 1e0
# MAX_PROBLEM_SIZE = 1e50
# MIN_YEAR = 2025
# MAX_YEAR = 2050

# # aggressive projection
# # constants in different format
# # fidelity_improvement_rate = .28
# # gate_speed_improvement_rate = .14
# # classical_speed_init = 1/(5*1e9) # seconds
# # superconducting_gate_speed_init = (1/(5*1e9))*(10**3.78) # seconds #overhead taken from quantum economic advantage calculator
# # initial_error = 10**(-2.5)
# # classical_speed_improvement_rate = 0.3
# # number_of_processors = 1e5
# # # time_upper_limt = 3.14*1e7 # 1-year of seconds
# # connectivity_penalty_exponent = 0.0 #connectivity penalty for physical to logical qubit ratio in this range.
# # time_upper_limit = (3.14*1e7)/12 # 1 week computation time
# # surface_code_overhead = 1e2 # number from Choi, Neil, and Moses
# # MAX_PROBLEM_SIZE = 1e50
# # MIN_YEAR = 2025
# # MAX_YEAR = 2050

# # default quantum and classical runtime if not specified
# classical_runtime = "n**3"
# quantum_runtime = "n"

# IBM_ROADMAP = {
#     2020: 27,
#     2022: 127,
#     2024: 133,
# }
# GOOGLE_ROADMAP = {
#     2019: 53,
#     2024: 105,
# }
# PERCENTILE_95_ROADMAP = {
#     2025: 1662,
#     2030: 15660,
# }
# SOTA_ROADMAP = {
#     2025: 1959,
#     2030: 24352,
# }
# percentile_90_roadmap = {2025: 305.13, 2030: 1315}

# default_roadmap = percentile_90_roadmap


# # gives the physical to logical overhead based on the surface code formula
# def surface_code_formula(pP: float) -> float:
#     pL = 1e-18
#     pth = 1e-2
#     numerator = 4 * math.log(math.sqrt(10 * pP / pL))
#     denominator = math.log(pth / pP)
#     fraction = numerator / denominator
#     f_QEC = (fraction + 1) ** -2
#     return f_QEC**-1


# #


# def problem_size_qubit_feasible(
#     roadmap: dict, year: int, alg_overhead: float = 1, q_prob_size="log"
# ) -> float:

#     # fit exponential to roadmap
#     years = np.array(list(roadmap.keys()))
#     qubits = np.array(list(roadmap.values()))
#     min_year = min(years)
#     # initial gues
#     p0 = [min(qubits), 0.5, 0]

#     # fit an exponential curve to the data
#     def exp_func(x, a, b, c):
#         return a * np.exp(b * (x - min_year)) + c

#     try:
#         popt, _ = curve_fit(
#             exp_func, years, qubits, p0=p0, bounds=([0, -2, -1000], [10000, 2, 1000])
#         )
#     except RuntimeError:
#         # print(f"Fitting failed for {label}")
#         return None
#     surf_overhead = surface_code_formula(
#         initial_error * (1-fidelity_improvement_rate) ** (year - 2025)
#     )
#     if q_prob_size == "log":
#         qubit_number = min(
#             exp_func(year, *popt) / (surf_overhead * alg_overhead_qubit),
#             np.log2(MAX_PROBLEM_SIZE),
#         )
#         return 2 ** (qubit_number)
#     else:
#         return min(
#             exp_func(year, *popt) / (surf_overhead * alg_overhead_qubit),
#             MAX_PROBLEM_SIZE,
#         )


# # quantum speed per operation function
# def quantum_seconds_per_operation(year):
#     gate_speed = superconducting_gate_speed_init * (
#         1 - gate_speed_improvement_rate
#     ) ** (year - 2025)
#     # with error correction
#     fidelity_year = initial_error * (1-fidelity_improvement_rate) ** (year - 2025)
#     proportional_change = (surface_code_formula(fidelity_year)) / surface_code_formula(
#         initial_error
#     )
#     return gate_speed * scode_init_speed_overhead * proportional_change ** (1.5)


# # seconds per effective operation ie here we factor in parallelism
# def classical_seconds_per_operation(year):
#     # just dividing by number of processors here for now to simplify things
#     return (
#         1e-9
#         * (1 - classical_speed_improvement_rate) ** (year - 2025)
#         / number_of_processors
#     )


# # maximum problem size that can be solved on a quantum computer in a given amount of time at a given year
# def find_largest_problem_size(
#     runtime_string,
#     year: int,
#     quantum=True,
#     qadv_only=False,
#     roadmap: dict = default_roadmap,
#     stagnation_year=2200,
#     time_upper_limit=time_upper_limit,
#     q_prob_size="log",
# ) -> float:
#     # Constants
#     try:
#         # Convert string expression to lambda function
#         n = sp.Symbol("n")
#         expr = sp.sympify(runtime_string)
#         expr = expr * n**connectivity_penalty_exponent * alg_overhead_qspeed
#         # Apply connectivity penalty to the runtime
#         runtime_func = sp.lambdify(n, expr)

#         if quantum:
#             quantum_total = lambda size: quantum_seconds_per_operation(
#                 year
#             ) * runtime_func(size)

#             def quantum_limit(x):
#                 return time_upper_limit  # number of seconds in a year
#                 # return q_ops_second_dollar

#             qadv = binary_search_intersection(
#                 quantum_total, quantum_limit, low=2.5, high=1e50
#             )
#             if qadv is None:
#                 # print(f"No intersection found between quantum total and limit for year {year}")
#                 qadv = float("inf")
#             # except Exception as e:
#             #     print(f"Error finding intersection for year {year}: {e}")
#             #     return float("inf")

#             size_feasible = problem_size_qubit_feasible(
#                 roadmap=roadmap, year=year, q_prob_size=q_prob_size
#             )
#             # Handle the case where size_feasible is None
#             if size_feasible is None:
#                 print(f"No feasible size found for year {year}. Returning infinity.")
#                 return float("inf")  # Return infinity if no feasible size is found
#             if not qadv_only:
#                 return min(qadv, size_feasible)
#             else:
#                 return qadv

#         else:

#             def classical_cost(x):
#                 if year < stagnation_year:
#                     return classical_seconds_per_operation(year) * runtime_func(x)
#                 else:
#                     return classical_seconds_per_operation(
#                         stagnation_year
#                     ) * runtime_func(x)

#             def classical_limit(x):
#                 return time_upper_limit
#                 # return ops_second_dollar

#             return binary_search_intersection(
#                 classical_cost, classical_limit, low=2.5, high=1e50
#             )

#     except Exception as e:
#         print(f"Error in function evaluation: {e}")
#         return None


# def quantum_advantage_size_by_year(
#     year, classical_runtime_string: str, quantum_runtime_string: str
# ) -> float:
#     n = sp.symbols("n")
#     class_expr = sp.sympify(classical_runtime_string)
#     class_expr = (
#         class_expr * classical_seconds_per_operation(year) * classical_alg_overhead
#     )
#     classical_runtime_func = sp.lambdify(n, class_expr)

#     quant_expr = sp.sympify(quantum_runtime_string)
#     quant_expr = quant_expr * n**connectivity_penalty_exponent * alg_overhead_qspeed
#     quant_expr = quant_expr * quantum_seconds_per_operation(year)
#     quantum_runtime_func = sp.lambdify(n, quant_expr)

#     # Find intersection using the original string expressions
#     return binary_search_intersection(
#         classical_runtime_func, quantum_runtime_func, 2.5, 1e50
#     )


# # this is the intersection of quantum advantage size by year and quantum problem size qubit feasible by year
# def generalized_qea(
#     classical_runtime_string: str, quantum_runtime_string: str
# ) -> float:
#     return binary_search_intersection(
#         lambda x: quantum_advantage_size_by_year(
#             x, classical_runtime_string, quantum_runtime_string
#         ),
#         lambda x: find_largest_problem_size(quantum_runtime_string, x, quantum=True),
#         MIN_YEAR,
#         MAX_YEAR,
#     )


# # %%
# runtime_pairs = [
#     ("n", "n**.5", "Grover's Algorithm"),
#     ("n", "log(n,2)", "Exponential Speedup"),
#     ("n**3", "n**2", "Matrix Multiplication"),
# ]


# # Modified get_intersection_year to accept roadmap parameter
# def get_intersection_year(
#     quantum_runtime,
#     classical_runtime,
#     roadmap=default_roadmap,
#     start_year=MIN_YEAR,
#     end_year=MAX_YEAR,
# ):
#     """Find intersection year for given runtime pair and roadmap"""

#     def find_largest_size(year):
#         return find_largest_problem_size(
#             quantum_runtime, year, quantum=True, roadmap=roadmap
#         )

#     def quantum_advantage_size(year):
#         return quantum_advantage_size_by_year(year, classical_runtime, quantum_runtime)

#     return binary_search_intersection(
#         find_largest_size, quantum_advantage_size, start_year, end_year
#     )


# def create_roadmap(growth_rate, init_qubits=100, start_year=2024, num_years=10):
#     return {
#         start_year + i: int(init_qubits * (growth_rate) ** i) for i in range(num_years)
#     }


# # Create a range of growth rates from 1.1 to 1.8
# growth_rates = np.arange(1.1, 1.9, 0.1)

# # Dictionary to store results for each algorithm
# intersection_results = {name: [] for _, _, name in runtime_pairs}

# # Calculate intersection years for each growth rate
# for growth_rate in growth_rates:
#     roadmap = create_roadmap(growth_rate)

#     for classical_rt, quantum_rt, name in runtime_pairs:
#         year = get_intersection_year(quantum_rt, classical_rt, roadmap)
#         intersection_results[name].append(year)

# # Create the plot
# plt.figure(figsize=(12, 8))

# # Plot lines for each algorithm
# for name in intersection_results.keys():
#     plt.plot(
#         growth_rates,
#         intersection_results[name],
#         "o-",
#         label=name,
#         linewidth=2,
#         markersize=8,
#     )

# plt.xlabel("Qubit Growth Rate (per year)", fontsize=12, fontweight="bold")
# plt.ylabel("Intersection Year", fontsize=12, fontweight="bold")
# plt.title(
#     "Quantum Advantage Intersection Year vs Qubit Growth Rate",
#     fontsize=14,
#     fontweight="bold",
# )
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=10)

# # Format x-axis to show percentages
# plt.xticks(growth_rates, [f"{(rate-1)*100:.0f}%" for rate in growth_rates])

# # Add horizontal line at 2050 to show upper limit
# plt.axhline(y=2050, color="r", linestyle="--", alpha=0.3, label="Year 2050")

# # Adjust layout to prevent label cutoff
# plt.tight_layout()

# # Save the figure
# plt.savefig(
#     "Figures/intersection_year_vs_growth_rate.png", dpi=300, bbox_inches="tight"
# )
# plt.show()

# # Print numerical results
# print("\nNumerical Results:")
# print("-" * 80)
# print(f"{'Growth Rate':<12}", end="")
# for name in intersection_results.keys():
#     print(f"{name:<22}", end="")
# print("\n" + "-" * 80)

# for i, rate in enumerate(growth_rates):
#     print(f"{(rate-1)*100:>3.0f}%{'':8}", end="")
#     for name in intersection_results.keys():
#         year = intersection_results[name][i]
#         year_str = f"{year:.1f}" if year is not None else "N/A"
#         print(f"{year_str:<22}", end="")
#     print()
# #%%


# # Create a range of fidelity improvement rates from 0.1 to 0.9
# fidelity_improvement_rates = np.arange(0.1, 1.0, 0.1)

# # Dictionary to store results for each algorithm
# intersection_results = {name: [] for _, _, name in runtime_pairs}

# # Store original fidelity improvement rate to restore later
# original_fidelity_rate = fidelity_improvement_rate

# # Calculate intersection years for each fidelity improvement rate
# for rate in fidelity_improvement_rates:
#     # Temporarily modify the global fidelity improvement rate
#     globals()['fidelity_improvement_rate'] = rate
    
#     for classical_rt, quantum_rt, name in runtime_pairs:
#         year = get_intersection_year(quantum_rt, classical_rt, start_year=2024, end_year=2051)
#         intersection_results[name].append(year)

# # Restore original value
# globals()['fidelity_improvement_rate'] = original_fidelity_rate

# # Create the plot
# plt.figure(figsize=(12, 8))

# # Plot lines for each algorithm
# for name in intersection_results.keys():
#     # Filter out None values for plotting
#     x_values = []
#     y_values = []
#     for i, year in enumerate(intersection_results[name]):
#         if year is not None:
#             x_values.append(fidelity_improvement_rates[i])
#             y_values.append(year)
    
#     if x_values:  # Only plot if there are valid points
#         plt.plot(x_values, y_values, "o-", label=name, linewidth=2, markersize=8)

# plt.xlabel("Fidelity Improvement Rate (per year)", fontsize=12, fontweight="bold")
# plt.ylabel("Intersection Year", fontsize=12, fontweight="bold")
# plt.title(
#     "Quantum Advantage Intersection Year vs Fidelity Improvement Rate",
#     fontsize=14,
#     fontweight="bold",
# )
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=10)

# # Format x-axis as percentages if desired
# plt.xticks(fidelity_improvement_rates, [f"{rate*100:.0f}%" for rate in fidelity_improvement_rates])

# # Add horizontal line at 2050 to show upper limit
# plt.axhline(y=2050, color="r", linestyle="--", alpha=0.3, label="Year 2050")

# # Adjust layout to prevent label cutoff
# plt.tight_layout()

# # Save the figure
# plt.savefig(
#     "Figures/intersection_year_vs_fidelity_rate.png", dpi=300, bbox_inches="tight"
# )
# plt.show()

# # Print numerical results
# print("\nNumerical Results:")
# print("-" * 80)
# print(f"{'Fidelity Rate':<15}", end="")
# for name in intersection_results.keys():
#     print(f"{name:<22}", end="")
# print("\n" + "-" * 80)

# for i, rate in enumerate(fidelity_improvement_rates):
#     print(f"{rate*100:>3.0f}%{'':11}", end="")
#     for name in intersection_results.keys():
#         year = intersection_results[name][i]
#         year_str = f"{year:.1f}" if year is not None else "N/A"
#         print(f"{year_str:<22}", end="")
#     print()
# #%%
# #Examining the Classical Improvement Rate

# # %%
# # Create a range of classical improvement rates from 0.1 to 0.9
# classical_improvement_rates = np.arange(0.1, 1.0, 0.1)

# # Dictionary to store results for each algorithm
# intersection_results = {name: [] for _, _, name in runtime_pairs}

# # Store original classical improvement rate to restore later
# original_classical_rate = classical_speed_improvement_rate

# # Calculate intersection years for each classical improvement rate
# for rate in classical_improvement_rates:
#     # Temporarily modify the global classical improvement rate
#     globals()['classical_speed_improvement_rate'] = rate
    
#     for classical_rt, quantum_rt, name in runtime_pairs:
#         year = get_intersection_year(quantum_rt, classical_rt, start_year=2024, end_year=2051)
#         intersection_results[name].append(year)

# # Restore original value
# globals()['classical_speed_improvement_rate'] = original_classical_rate

# # Create the plot
# plt.figure(figsize=(12, 8))

# # Plot lines for each algorithm
# for name in intersection_results.keys():
#     # Filter out None values for plotting
#     x_values = []
#     y_values = []
#     for i, year in enumerate(intersection_results[name]):
#         if year is not None:
#             x_values.append(classical_improvement_rates[i])
#             y_values.append(year)
    
#     if x_values:  # Only plot if there are valid points
#         plt.plot(x_values, y_values, "o-", label=name, linewidth=2, markersize=8)

# plt.xlabel("Classical Improvement Rate (per year)", fontsize=12, fontweight="bold")
# plt.ylabel("Intersection Year", fontsize=12, fontweight="bold")
# plt.title(
#     "Quantum Advantage Intersection Year vs Classical Improvement Rate",
#     fontsize=14,
#     fontweight="bold",
# )
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=10)

# # Format x-axis as percentages if desired
# plt.xticks(classical_improvement_rates, [f"{rate*100:.0f}%" for rate in classical_improvement_rates])

# # Add horizontal line at 2050 to show upper limit
# plt.axhline(y=2050, color="r", linestyle="--", alpha=0.3, label="Year 2050")

# # Adjust layout to prevent label cutoff
# plt.tight_layout()

# # Save the figure
# plt.savefig(
#     "Figures/intersection_year_vs_classical_rate.png", dpi=300, bbox_inches="tight"
# )
# plt.show()

# # Print numerical results
# print("\nNumerical Results:")
# print("-" * 80)
# print(f"{'Classical Rate':<15}", end="")
# for name in intersection_results.keys():
#     print(f"{name:<22}", end="")
# print("\n" + "-" * 80)

# for i, rate in enumerate(classical_improvement_rates):
#     print(f"{rate*100:>3.0f}%{'':11}", end="")
#     for name in intersection_results.keys():
#         year = intersection_results[name][i]
#         year_str = f"{year:.1f}" if year is not None else "N/A"
#         print(f"{year_str:<22}", end="")
#     print()















# # %%
# # Analyze intersection years for different classical improvement rates and algorithm types




# # error_reduction_rate = .2
# # gate_speed_improvement_rate = .15
# # classical_speed_init = 1/(5*1e9) # seconds
# # superconducting_gate_speed_init = (1/(5*1e9))*(10**3.78) # seconds
# # initial_error = 10**(-2.5)
# # classical_speed_improvement_rate = .3
# # number_of_processors = 1e5

# # #constants in different format
# fidelity_improvement_rate = 0.28
# gate_speed_improvement_rate = 0.14
# classical_speed_init = 1 / (1.5 * 1e9)  #
# superconducting_gate_speed_init = 1e-6  # (1/(1.5*1e9))*(10**3.78) # seconds #overhead taken from quantum economic advantage calculator
# initial_error = 10 ** (-2.5)
# classical_speed_improvement_rate = 0.3  # moore's law improvement
# number_of_processors = 1e8  # processor overehead done from calculations for GPUs
# # time_upper_limt = 3.14*1e7 # 1-year of seconds
# connectivity_penalty_exponent = 0.0  # connectivity penalty for physical to logical qubit ratio in this range. default no asymptotic connectivity penality
# time_upper_limit = 4 * (3.14 * 1e7) / 52  # 1 month computation time
# scode_init_speed_overhead = 1e2  # slowdown overhead from Choi, Neil, and Moses
# alg_overhead_qubit = 1e1  # algorithm overhead in logical qubits
# alg_overhead_qspeed = 1e0  # algorithm speed overhead based on constants this is exclusivly for quantum algorithm
# classical_alg_overhead = 1e0
# MAX_PROBLEM_SIZE = 1e50
# MIN_YEAR = 2025
# MAX_YEAR = 2050

# constant_variations = [10**i for i in range(-3, 5, 1)]


# # First runtime pair: n vs n^0.5
# # Define runtime pairs
# runtime_pairs = [
#     ("n", "n**.5", "Grover's Algorithm"),
#     ("n", "log(n,2)", "Exponential Speedup"),
#     ("n**3", "n**2", "Matrix Multiplication"),
# ]
# # Get initial intersection years
# initial_years = []
# for classical_rt, quantum_rt, name in runtime_pairs:
#     year = get_intersection_year(quantum_rt, classical_rt)
#     initial_years.append(year)
#     print(f"Initial intersection year for {name}: {year}")

# # Analyze sensitivity to error reduction rate
# classical_improvement_rates = np.arange(0.1, 0.9, 0.1)
# intersection_results_quantum_constants = {name: [] for _, _, name in runtime_pairs}

# for constant_variation in constant_variations:
#     superconducting_gate_speed_init = (
#         superconducting_gate_speed_init * constant_variation
#     )
#     for classical_rt, quantum_rt, name in runtime_pairs:
#         year = get_intersection_year(
#             quantum_rt, classical_rt, start_year=2024, end_year=2051
#         )
#         intersection_results_quantum_constants[name].append(year)


# # %%
# plt.plot(
#     constant_variations,
#     intersection_results_quantum_constants["Grover's Algorithm"],
#     label="Grover's Algorithm",
# )
# plt.scatter(
#     constant_variations, intersection_results_quantum_constants["Grover's Algorithm"]
# )
# plt.plot(
#     constant_variations,
#     intersection_results_quantum_constants["Exponential Speedup"],
#     label="Exponential Speedup",
# )
# plt.scatter(
#     constant_variations, intersection_results_quantum_constants["Exponential Speedup"]
# )
# plt.plot(
#     constant_variations,
#     intersection_results_quantum_constants["Matrix Multiplication"],
#     label="Matrix Multiplication",
# )
# plt.scatter(
#     constant_variations, intersection_results_quantum_constants["Matrix Multiplication"]
# )
# plt.ylabel("Generalized QEA Intersection Year")
# plt.xlabel("Quantum Constants")
# plt.xscale("log")
# plt.grid(True)
# plt.legend(loc="upper right")
# plt.savefig("Figures/intersection_year_vs_classical_improvement_rate.png")
# plt.title("Intersection Year vs Quantum Constants for Different Runtime Complexities")

# # %% [markdown]
# # # Sensitivity Error Reduction


# # %%
# # First runtime pair: n vs n^0.5
# # Define runtime pairs
# runtime_pairs = [
#     ("n", "n**.5", "Grover's Algorithm"),
#     ("n", "log(n,2)", "Exponential Speedup"),
#     ("n**3", "n**2", "Matrix Multiplication"),
# ]


# def get_intersection_year(
#     quantum_runtime, classical_runtime, start_year=MIN_YEAR, end_year=MAX_YEAR
# ):
#     """Find intersection year for given runtime pair"""

#     def find_largest_size(year):
#         return find_largest_problem_size(quantum_runtime, year, quantum=True)

#     def quantum_advantage_size(year):
#         return quantum_advantage_size_by_year(year, classical_runtime, quantum_runtime)

#     return binary_search_intersection(
#         find_largest_size, quantum_advantage_size, start_year, end_year
#     )


# # Get initial intersection years
# initial_years = []
# for classical_rt, quantum_rt, name in runtime_pairs:
#     year = get_intersection_year(quantum_rt, classical_rt)
#     initial_years.append(year)
#     print(f"Initial intersection year for {name}: {year}")

# # Analyze sensitivity to error reduction rate
# error_reduction_ranges = np.arange(0.1, 0.9, 0.1)
# intersection_results_error_reduction = {name: [] for _, _, name in runtime_pairs}

# for error_rate in error_reduction_ranges:
#     fidelity_improvement_rate = error_rate
#     for classical_rt, quantum_rt, name in runtime_pairs:
#         year = get_intersection_year(
#             quantum_rt, classical_rt, start_year=2024, end_year=2051
#         )
#         intersection_results_error_reduction[name].append(year)


# # %%
# # now plot the intersection years vs error reduction
# plt.plot(
#     error_reduction_ranges * 100,
#     intersection_results_error_reduction["Grover's Algorithm"],
#     label="grovers algorithm",
# )
# plt.scatter(
#     error_reduction_ranges * 100,
#     intersection_results_error_reduction["Grover's Algorithm"],
# )
# plt.plot(
#     error_reduction_ranges * 100,
#     intersection_results_error_reduction["Exponential Speedup"],
#     label="exponential speedup",
# )
# plt.scatter(
#     error_reduction_ranges * 100,
#     intersection_results_error_reduction["Exponential Speedup"],
# )
# plt.plot(
#     error_reduction_ranges * 100,
#     intersection_results_error_reduction["Matrix Multiplication"],
#     label="matrix multiplication",
# )
# plt.scatter(
#     error_reduction_ranges * 100,
#     intersection_results_error_reduction["Matrix Multiplication"],
# )
# plt.ylabel("Generalized QEA Intersection Year")
# plt.xlabel("Percent Decrease in Gate Error Per Year")
# plt.title("Intersection Year vs Gate Error Reduction Rate")
# plt.legend(loc="upper right")
# plt.grid(True)
# plt.savefig("Figures/intersection_year_vs_error_reduction_rate.png")
# plt.show()


# # %%
# # find the intersection of quantum max computable problem size and quantum economic advantage use binary search intersection


# # First runtime pair: n vs n^0.5
# # Define runtime pairs
# runtime_pairs = [
#     ("n", "n**.5", "Grover's Algorithm"),
#     ("n", "log(n,2)", "Exponential Speedup"),
#     ("n**3", "n**2", "Matrix Multiplication"),
# ]
# # Get initial intersection years
# initial_years = []
# for classical_rt, quantum_rt, name in runtime_pairs:
#     year = get_intersection_year(quantum_rt, classical_rt)
#     initial_years.append(year)
#     print(f"Initial intersection year for {name}: {year}")

# # Analyze sensitivity to error reduction rate
# speed_improvement_ranges = np.arange(0.1, 0.9, 0.1)
# intersection_results_speed_improvement = {name: [] for _, _, name in runtime_pairs}

# for speed_improvement in speed_improvement_ranges:
#     gate_speed_improvement_rate = speed_improvement
#     for classical_rt, quantum_rt, name in runtime_pairs:
#         year = get_intersection_year(
#             quantum_rt, classical_rt, start_year=2024, end_year=2051
#         )
#         intersection_results_speed_improvement[name].append(year)


# # %%
# plt.plot(
#     speed_improvement_ranges * 100,
#     intersection_results_speed_improvement["Grover's Algorithm"],
#     label="Grover's Algorithm",
# )
# plt.scatter(
#     speed_improvement_ranges * 100,
#     intersection_results_speed_improvement["Grover's Algorithm"],
# )
# plt.plot(
#     speed_improvement_ranges * 100,
#     intersection_results_speed_improvement["Exponential Speedup"],
#     label="Exponential Speedup",
# )
# plt.scatter(
#     speed_improvement_ranges * 100,
#     intersection_results_speed_improvement["Exponential Speedup"],
# )
# plt.plot(
#     speed_improvement_ranges * 100,
#     intersection_results_speed_improvement["Matrix Multiplication"],
#     label="Matrix Multiplication",
# )
# plt.scatter(
#     speed_improvement_ranges * 100,
#     intersection_results_speed_improvement["Matrix Multiplication"],
# )
# plt.ylabel("Generalized QEA Intersection Year")
# plt.xlabel("Percent Decrease in Quantum Gate Time Per Year")
# plt.title("Intersection Year vs Gate Speed Improvement Rate")
# plt.grid(True)
# plt.legend(loc="upper right")
# plt.savefig("Figures/intersection_year_vs_gate_speed_improvement_rate.png")
# plt.show()
# # %%

# # %% [markdown]
# # # Sensitivity Classical Improvement Rate

# # %%
# # Get initial intersection years
# initial_years = []
# for classical_rt, quantum_rt, name in runtime_pairs:
#     year = get_intersection_year(quantum_rt, classical_rt)
#     initial_years.append(year)
#     print(f"Initial intersection year for {name}: {year}")

# # Analyze sensitivity to error reduction rate
# classical_improvement_rates = np.arange(0.1, 0.9, 0.1)
# intersection_results_classical_improvement = {name: [] for _, _, name in runtime_pairs}

# for classical_improvement_rate in classical_improvement_rates:
#     classical_speed_improvement_rate = classical_improvement_rate
#     for classical_rt, quantum_rt, name in runtime_pairs:
#         year = get_intersection_year(
#             quantum_rt, classical_rt, start_year=2024, end_year=2051
#         )
#         intersection_results_classical_improvement[name].append(year)
# # %%

# print(intersection_results_classical_improvement)

# #%%

# plt.plot(
#     classical_improvement_rates * 100,
#     intersection_results_classical_improvement["Grover's Algorithm"],
#     label="Grover's Algorithm",
# )
# plt.scatter(
#     classical_improvement_rates * 100,
#     intersection_results_classical_improvement["Grover's Algorithm"],
# )
# plt.plot(
#     classical_improvement_rates * 100,
#     intersection_results_classical_improvement["Exponential Speedup"],
#     label="Exponential Speedup",
# )
# plt.scatter(
#     classical_improvement_rates * 100,
#     intersection_results_classical_improvement["Exponential Speedup"],
# )
# plt.plot(
#     classical_improvement_rates * 100,
#     intersection_results_classical_improvement["Matrix Multiplication"],
#     label="Matrix Multiplication",
# )
# plt.scatter(
#     classical_improvement_rates * 100,
#     intersection_results_classical_improvement["Matrix Multiplication"],
# )
# plt.ylabel("Generalized QEA Intersection Year")
# plt.xlabel("Classical Improvement Rate Per Year")
# plt.grid(True)
# plt.legend(loc="upper right")
# plt.savefig("Figures/intersection_year_vs_classical_improvement_rate.png")
# plt.title(
#     "Intersection Year vs Classical Improvement Rate for Different Runtime Complexities"
# )
# # %%
