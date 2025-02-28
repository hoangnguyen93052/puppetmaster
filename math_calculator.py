import math
import statistics
from typing import List, Tuple

class BasicMath:
    @staticmethod
    def add(x: float, y: float) -> float:
        return x + y

    @staticmethod
    def subtract(x: float, y: float) -> float:
        return x - y

    @staticmethod
    def multiply(x: float, y: float) -> float:
        return x * y

    @staticmethod
    def divide(x: float, y: float) -> float:
        if y == 0:
            raise ValueError("Cannot divide by zero.")
        return x / y

class Algebra:
    @staticmethod
    def quadratic_roots(a: float, b: float, c: float) -> Tuple[complex, complex]:
        discriminant = b**2 - 4*a*c
        root1 = (-b + cmath.sqrt(discriminant)) / (2*a)
        root2 = (-b - cmath.sqrt(discriminant)) / (2*a)
        return root1, root2

    @staticmethod
    def factorial(n: int) -> int:
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers.")
        return 1 if n == 0 else n * Algebra.factorial(n - 1)

class Statistics:
    @staticmethod
    def mean(data: List[float]) -> float:
        return statistics.mean(data)

    @staticmethod
    def median(data: List[float]) -> float:
        return statistics.median(data)

    @staticmethod
    def mode(data: List[float]) -> float:
        return statistics.mode(data)

    @staticmethod
    def variance(data: List[float]) -> float:
        return statistics.variance(data)

    @staticmethod
    def standard_deviation(data: List[float]) -> float:
        return statistics.stdev(data)

class Calculus:
    @staticmethod
    def differentiate(func, x: float, h: float = 1e-5) -> float:
        return (func(x + h) - func(x)) / h

    @staticmethod
    def integrate(func, a: float, b: float, n: int = 1000) -> float:
        step = (b - a) / n
        return sum(func(a + i * step) * step for i in range(n))

class MathCalculator:
    @staticmethod
    def basic_operations(x: float, y: float):
        print("Basic Operations:")
        print(f"Addition: {BasicMath.add(x, y)}")
        print(f"Subtraction: {BasicMath.subtract(x, y)}")
        print(f"Multiplication: {BasicMath.multiply(x, y)}")
        print(f"Division: {BasicMath.divide(x, y)}")

    @staticmethod
    def polynomial_roots(a: float, b: float, c: float):
        roots = Algebra.quadratic_roots(a, b, c)
        print(f"Roots of the polynomial {a}x² + {b}x + {c} are: {roots}")

    @staticmethod
    def calculate_statistics(data: List[float]):
        print("Statistics:")
        print(f"Mean: {Statistics.mean(data)}")
        print(f"Median: {Statistics.median(data)}")
        print(f"Mode: {Statistics.mode(data)}")
        print(f"Variance: {Statistics.variance(data)}")
        print(f"Standard Deviation: {Statistics.standard_deviation(data)}")

    @staticmethod
    def differentiate_function(func, x: float):
        derivative = Calculus.differentiate(func, x)
        print(f"The derivative of the function at x={x} is: {derivative}")
    
    @staticmethod
    def integrate_function(func, a: float, b: float):
        integral = Calculus.integrate(func, a, b)
        print(f"The integral of the function from {a} to {b} is: {integral}")

def sample_function(x):
    return x**2 + 3*x + 2

if __name__ == "__main__":
    math_calc = MathCalculator()

    # Basic operations
    math_calc.basic_operations(10, 5)

    # Polynomial roots
    math_calc.polynomial_roots(1, -3, 2)

    # Statistics
    data_points = [1, 2, 2, 3, 4, 5, 5, 5, 6, 7]
    math_calc.calculate_statistics(data_points)

    # Calculus
    math_calc.differentiate_function(sample_function, 2)
    math_calc.integrate_function(sample_function, 0, 2)

    # Further calculations
    print("\nFurther calculations:")
    print(f"Factorial of 5: {Algebra.factorial(5)}")