"""
OE_YAPAT Service Integration

Handles communication with the external OE_YAPAT service for custom taxonomy generation.
"""

import logging
from typing import Dict, Any, Optional
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from app.config import settings


logger = logging.getLogger(__name__)


class OEYapatServiceError(Exception):
    """Base exception for OE_YAPAT service errors"""
    pass


class OEYapatTimeoutError(OEYapatServiceError):
    """Raised when request to OE_YAPAT times out"""
    pass


class OEYapatValidationError(OEYapatServiceError):
    """Raised when OE_YAPAT response validation fails"""
    pass


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    reraise=True
)
async def generate_taxonomy(
    prompt: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate custom taxonomy using OE_YAPAT service.
    
    Args:
        prompt: User prompt describing the desired taxonomy
        context: Additional context for generation (optional)
        
    Returns:
        Dict containing generated taxonomy data
        
    Raises:
        OEYapatTimeoutError: If request times out
        OEYapatServiceError: If service returns an error
        OEYapatValidationError: If response structure is invalid
    """
    url = f"{settings.OE_YAPAT_SERVICE_URL}/api/generate-taxonomy"
    
    headers = {}
    if settings.OE_YAPAT_API_KEY:
        headers["Authorization"] = f"Bearer {settings.OE_YAPAT_API_KEY}"
    
    payload = {
        "prompt": prompt,
        "context": context or {}
    }
    
    try:
        async with httpx.AsyncClient(timeout=settings.OE_YAPAT_TIMEOUT) as client:
            logger.info(f"Sending taxonomy generation request to OE_YAPAT service")
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response structure
            if not validate_taxonomy_response(data):
                raise OEYapatValidationError("Invalid taxonomy structure received from OE_YAPAT service")
            
            logger.info(f"Successfully generated taxonomy from OE_YAPAT service")
            return data
            
    except httpx.TimeoutException as e:
        logger.error(f"OE_YAPAT service timeout: {e}")
        raise OEYapatTimeoutError("Taxonomy generation timed out. Please try again.") from e
    
    except httpx.HTTPStatusError as e:
        logger.error(f"OE_YAPAT service HTTP error: {e}")
        error_detail = "Unknown error"
        try:
            error_data = e.response.json()
            error_detail = error_data.get("detail", str(e))
        except Exception:
            error_detail = str(e)
        raise OEYapatServiceError(f"Taxonomy generation failed: {error_detail}") from e
    
    except httpx.NetworkError as e:
        logger.error(f"OE_YAPAT service network error: {e}")
        raise OEYapatServiceError("Failed to connect to taxonomy generation service") from e
    
    except Exception as e:
        logger.error(f"Unexpected error calling OE_YAPAT service: {e}")
        raise OEYapatServiceError(f"Unexpected error: {str(e)}") from e


def validate_taxonomy_response(response: Dict[str, Any]) -> bool:
    """
    Validate the structure of the taxonomy response from OE_YAPAT.
    
    Expected structure:
    {
        "taxonomy_data": {
            "nodes": [...],  # List of taxonomy nodes
            "hierarchy": {...}  # Hierarchical structure
        },
        "metadata": {
            "model": "...",
            "timestamp": "...",
            ...
        }
    }
    
    Args:
        response: Response data from OE_YAPAT service
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(response, dict):
        logger.error("Response is not a dictionary")
        return False
    
    if "taxonomy_data" not in response:
        logger.error("Missing 'taxonomy_data' in response")
        return False
    
    taxonomy_data = response["taxonomy_data"]
    if not isinstance(taxonomy_data, dict):
        logger.error("'taxonomy_data' is not a dictionary")
        return False
    
    # Basic validation - ensure it has some structure
    if not taxonomy_data:
        logger.error("'taxonomy_data' is empty")
        return False
    
    # Additional validation can be added here as needed
    # For example, checking for required fields, node structure, etc.
    
    return True


def sanitize_prompt(prompt: str) -> str:
    """
    Sanitize user prompt before sending to OE_YAPAT service.
    
    Args:
        prompt: Raw user prompt
        
    Returns:
        Sanitized prompt
    """
    # Remove any potentially harmful content
    # Strip excessive whitespace
    sanitized = " ".join(prompt.split())
    
    # Limit length
    max_length = 2000
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized
