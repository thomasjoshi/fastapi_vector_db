"""
Exceptions used by services.

This module centralizes all exception definitions used throughout the application,
ensuring consistent error handling and reporting.
"""
from typing import Any, Optional


class ServiceError(Exception):
    """Base class for all service-level exceptions."""

    pass


class NotFoundError(ServiceError):
    """
    Raised when a resource is not found.

    This exception includes information about the resource type and ID
    to facilitate better error reporting.
    """

    def __init__(
        self,
        message: str,
        resource_type: str = "Resource",
        resource_id: Optional[Any] = None,
    ) -> None:
        """
        Initialize a NotFoundError.

        Args:
            message: Human-readable error message
            resource_type: Type of resource that was not found (e.g., "Library", "Document")
            resource_id: ID of the resource not found
        """
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(message)


class ValidationError(ServiceError):
    """
    Raised when validation fails.

    This is used for business rule validations that aren't covered by Pydantic.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize a ValidationError.

        Args:
            message: Human-readable error message
        """
        super().__init__(message)


class DuplicateResourceError(ServiceError):
    """
    Raised when a resource with the same ID already exists.

    This exception includes information about the resource type
    to facilitate better error reporting.
    """

    def __init__(self, message: str, resource_type: str = "Resource") -> None:
        """
        Initialize a DuplicateResourceError.

        Args:
            message: Human-readable error message
            resource_type: Type of resource that was duplicated
        """
        self.resource_type = resource_type
        super().__init__(message)
