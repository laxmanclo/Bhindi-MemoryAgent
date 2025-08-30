/**
 * Simple calculator service for demonstration purposes
 */
export class CalculatorService {
  /**
   * Add two numbers
   * @param a First number
   * @param b Second number
   * @returns Sum of the two numbers
   */
  add(a: number, b: number): number {
    return a + b;
  }

  /**
   * Subtract second number from first number
   * @param a Number to subtract from
   * @param b Number to subtract
   * @returns Result of subtraction
   */
  subtract(a: number, b: number): number {
    return a - b;
  }

  /**
   * Multiply two numbers
   * @param a First number
   * @param b Second number
   * @returns Product of the two numbers
   */
  multiply(a: number, b: number): number {
    return a * b;
  }

  /**
   * Divide first number by second number
   * @param a Dividend (number to be divided)
   * @param b Divisor (number to divide by)
   * @returns Result of division
   * @throws Error if dividing by zero
   */
  divide(a: number, b: number): number {
    if (b === 0) {
      throw new Error('Cannot divide by zero');
    }
    return a / b;
  }

  /**
   * Calculate a number raised to a power
   * @param base Base number
   * @param exponent Exponent
   * @returns Result of raising base to the power of exponent
   */
  power(base: number, exponent: number): number {
    return Math.pow(base, exponent);
  }

  /**
   * Calculate the square root of a number
   * @param number Number to calculate square root of
   * @returns Square root of the number
   * @throws Error if input is negative
   */
  sqrt(number: number): number {
    if (number < 0) {
      throw new Error('Cannot calculate square root of a negative number');
    }
    return Math.sqrt(number);
  }

  /**
   * Calculate percentage of a number
   * @param percentage Percentage value
   * @param of Number to calculate percentage of
   * @returns Percentage of the number
   */
  percentage(percentage: number, of: number): number {
    return (percentage / 100) * of;
  }

  /**
   * Calculate factorial of a number
   * @param number Number to calculate factorial of
   * @returns Factorial of the number
   * @throws Error if input is negative
   */
  factorial(number: number): number {
    if (number < 0) {
      throw new Error('Cannot calculate factorial of a negative number');
    }
    if (number > 170) {
      throw new Error('Number too large for factorial calculation');
    }
    if (number === 0 || number === 1) {
      return 1;
    }
    let result = 1;
    for (let i = 2; i <= number; i++) {
      result *= i;
    }
    return result;
  }

  /**
   * Count occurrences of a character in a text
   * @param character Character to count
   * @param text Text to search in
   * @returns Number of occurrences
   */
  countCharacter(character: string, text: string): number {
    if (character.length !== 1) {
      throw new Error('Character must be a single character');
    }
    return (text.match(new RegExp(character, 'g')) || []).length;
  }
}
