"""
Custom exceptions for the service layer.
These provide more specific error types than generic exceptions.
"""


class ServiceError(Exception):
    """Base class for all service-level exceptions."""

    pass


class NotFoundError(ServiceError):
    """Raised when an entity is not found in the repository."""

    def __init__(self, entity_type: str, entity_id: str, message: str = None):
        self.entity_type = entity_type
        self.entity_id = entity_id
        # Also set resource_type and resource_id for the centralized error handler
        self.resource_type = entity_type
        self.resource_id = entity_id
        self.message = message or f"{entity_type} with ID {entity_id} not found"
        super().__init__(self.message)
