/**
 * Agent response types
 */

/**
 * Base success response DTO
 */
export class BaseSuccessResponseDto {
  success: boolean;
  responseType: string;
  data: any;

  /**
   * Constructor for BaseSuccessResponseDto
   * @param data Response data
   * @param responseType Type of response (text, mixed, etc.)
   */
  constructor(data: any, responseType: 'text' | 'mixed' = 'text') {
    this.success = true;
    this.responseType = responseType;
    this.data = responseType === 'text' 
      ? { text: typeof data === 'string' ? data : JSON.stringify(data) } 
      : data;
  }
}

/**
 * Base error response DTO
 */
export class BaseErrorResponseDto {
  success: boolean;
  error: {
    message: string;
    code: number;
    details?: string;
  };

  /**
   * Constructor for BaseErrorResponseDto
   * @param message Error message
   * @param code HTTP status code
   * @param details Additional error details
   */
  constructor(message: string, code: number = 500, details?: string) {
    this.success = false;
    this.error = {
      message,
      code,
      details
    };
  }
}
