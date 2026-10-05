"""Access-control primitives for filtering document and vector visibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Set


CLASSIFICATION_LEVELS = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}


@dataclass(frozen=True)
class AccessContext:
    """The caller identity and clearance used for document visibility checks."""

    tenant_id: str
    user_id: Optional[str] = None
    roles: Set[str] = field(default_factory=set)
    departments: Set[str] = field(default_factory=set)
    clearance_level: str = "public"

    def __post_init__(self) -> None:
        if self.clearance_level.lower() not in CLASSIFICATION_LEVELS:
            raise ValueError(f"Unknown clearance level: {self.clearance_level}")


class AccessPolicy:
    """Secure-by-default policy for normalized metadata records."""

    def can_access(self, metadata: Mapping[str, Any], context: AccessContext) -> bool:
        if metadata.get("tenant_id") != context.tenant_id:
            return False

        classification = str(metadata.get("classification", "public")).lower()
        if classification not in CLASSIFICATION_LEVELS:
            return False
        if CLASSIFICATION_LEVELS[classification] > CLASSIFICATION_LEVELS[context.clearance_level.lower()]:
            return False

        if not self._matches_acl(metadata, context):
            return False
        return True

    def filter_metadata(self, metadata_items: Iterable[Mapping[str, Any]], context: AccessContext) -> list[Mapping[str, Any]]:
        return [metadata for metadata in metadata_items if self.can_access(metadata, context)]

    def _matches_acl(self, metadata: Mapping[str, Any], context: AccessContext) -> bool:
        acl = metadata.get("acl")
        acl = acl if isinstance(acl, Mapping) else {}
        allowed_users = self._values(metadata, acl, "allowed_user_ids", "users")
        allowed_roles = self._values(metadata, acl, "allowed_roles", "roles")
        allowed_departments = self._values(metadata, acl, "allowed_departments", "departments")

        if allowed_users and context.user_id not in allowed_users:
            return False
        if allowed_roles and not context.roles.intersection(allowed_roles):
            return False
        if allowed_departments and not context.departments.intersection(allowed_departments):
            return False

        owner_user_id = metadata.get("owner_user_id")
        if owner_user_id and owner_user_id != context.user_id and not context.roles.intersection({"admin", "owner"}):
            return False
        return True

    def _values(self, metadata: Mapping[str, Any], acl: Mapping[str, Any], *keys: str) -> Set[str]:
        values = []
        for key in keys:
            value = metadata.get(key, acl.get(key))
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, (list, tuple, set)):
                values.extend(value)
        return {str(value) for value in values if value}


__all__ = ["AccessContext", "AccessPolicy", "CLASSIFICATION_LEVELS"]