"""Real deployment control service for managing Render services."""
from __future__ import annotations

import httpx
import asyncio
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncpg

class DeploymentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"

class ActionType(Enum):
    RESTART = "restart"
    DEPLOY = "deploy"
    SCALE = "scale"
    SUSPEND = "suspend"
    RESUME = "resume"

@dataclass
class DeploymentAction:
    id: int
    user_id: int
    action_type: ActionType
    target_service: str
    parameters: Dict[str, Any]
    status: DeploymentStatus
    result: Optional[Dict[str, Any]]
    started_at: datetime
    completed_at: Optional[datetime]

class DeploymentService:
    """Production deployment service with real Render API integration."""
    
    def __init__(self, postgres_dsn: str, render_api_key: Optional[str] = None):
        self.postgres_dsn = postgres_dsn
        self.render_api_key = render_api_key
        self.render_base_url = "https://api.render.com/v1"
        
    async def create_deployment_action(
        self,
        user_id: int,
        action_type: str,
        target_service: str,
        parameters: Dict[str, Any] = None
    ) -> DeploymentAction:
        """Create a new deployment action."""
        parameters = parameters or {}
        
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            result = await conn.fetchrow("""
                INSERT INTO deployment_actions (user_id, action_type, target_service, parameters, status, started_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                RETURNING id, user_id, action_type, target_service, parameters, status, result, started_at, completed_at
            """, user_id, action_type, target_service, parameters, DeploymentStatus.PENDING.value)
            
            action = DeploymentAction(
                id=result["id"],
                user_id=result["user_id"],
                action_type=ActionType(result["action_type"]),
                target_service=result["target_service"],
                parameters=result["parameters"],
                status=DeploymentStatus(result["status"]),
                result=result["result"],
                started_at=result["started_at"],
                completed_at=result["completed_at"]
            )
            
            # Execute the action asynchronously
            asyncio.create_task(self._execute_action(action))
            
            return action
        finally:
            await conn.close()
    
    async def _execute_action(self, action: DeploymentAction):
        """Execute a deployment action via Render API."""
        if not self.render_api_key:
            await self._update_action_status(
                action.id,
                DeploymentStatus.FAILED,
                {"error": "Render API key not configured"}
            )
            return
        
        try:
            await self._update_action_status(action.id, DeploymentStatus.RUNNING, {})
            
            headers = {
                "Authorization": f"Bearer {self.render_api_key}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                if action.action_type == ActionType.RESTART:
                    result = await self._restart_service(client, headers, action.target_service)
                
                elif action.action_type == ActionType.DEPLOY:
                    result = await self._deploy_service(client, headers, action.target_service, action.parameters)
                
                elif action.action_type == ActionType.SCALE:
                    result = await self._scale_service(client, headers, action.target_service, action.parameters)
                
                elif action.action_type == ActionType.SUSPEND:
                    result = await self._suspend_service(client, headers, action.target_service)
                
                elif action.action_type == ActionType.RESUME:
                    result = await self._resume_service(client, headers, action.target_service)
                
                else:
                    result = {"error": f"Unknown action type: {action.action_type}"}
                
                status = DeploymentStatus.COMPLETED if "error" not in result else DeploymentStatus.FAILED
                await self._update_action_status(action.id, status, result)
                
        except Exception as e:
            await self._update_action_status(
                action.id,
                DeploymentStatus.FAILED,
                {"error": f"Execution failed: {str(e)}"}
            )
    
    async def _restart_service(self, client: httpx.AsyncClient, headers: Dict[str, str], service_id: str) -> Dict[str, Any]:
        """Restart a Render service."""
        response = await client.post(
            f"{self.render_base_url}/services/{service_id}/restart",
            headers=headers
        )
        
        if response.status_code == 200:
            return {
                "action": "restart",
                "service_id": service_id,
                "status": "initiated",
                "response": response.json()
            }
        else:
            return {
                "error": f"Restart failed with status {response.status_code}",
                "response": response.text
            }
    
    async def _deploy_service(self, client: httpx.AsyncClient, headers: Dict[str, str], service_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy a Render service."""
        deploy_payload = {
            "clearCache": parameters.get("clear_cache", False)
        }
        
        response = await client.post(
            f"{self.render_base_url}/services/{service_id}/deploys",
            headers=headers,
            json=deploy_payload
        )
        
        if response.status_code == 201:
            deploy_data = response.json()
            return {
                "action": "deploy",
                "service_id": service_id,
                "deploy_id": deploy_data.get("id"),
                "status": deploy_data.get("status"),
                "response": deploy_data
            }
        else:
            return {
                "error": f"Deploy failed with status {response.status_code}",
                "response": response.text
            }
    
    async def _scale_service(self, client: httpx.AsyncClient, headers: Dict[str, str], service_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Scale a Render service."""
        scale_payload = {
            "numInstances": parameters.get("instances", 1)
        }
        
        response = await client.patch(
            f"{self.render_base_url}/services/{service_id}",
            headers=headers,
            json=scale_payload
        )
        
        if response.status_code == 200:
            return {
                "action": "scale",
                "service_id": service_id,
                "instances": parameters.get("instances"),
                "status": "updated",
                "response": response.json()
            }
        else:
            return {
                "error": f"Scale failed with status {response.status_code}",
                "response": response.text
            }
    
    async def _suspend_service(self, client: httpx.AsyncClient, headers: Dict[str, str], service_id: str) -> Dict[str, Any]:
        """Suspend a Render service."""
        suspend_payload = {
            "suspended": "suspend"
        }
        
        response = await client.patch(
            f"{self.render_base_url}/services/{service_id}",
            headers=headers,
            json=suspend_payload
        )
        
        if response.status_code == 200:
            return {
                "action": "suspend",
                "service_id": service_id,
                "status": "suspended",
                "response": response.json()
            }
        else:
            return {
                "error": f"Suspend failed with status {response.status_code}",
                "response": response.text
            }
    
    async def _resume_service(self, client: httpx.AsyncClient, headers: Dict[str, str], service_id: str) -> Dict[str, Any]:
        """Resume a Render service."""
        resume_payload = {
            "suspended": "not_suspended"
        }
        
        response = await client.patch(
            f"{self.render_base_url}/services/{service_id}",
            headers=headers,
            json=resume_payload
        )
        
        if response.status_code == 200:
            return {
                "action": "resume",
                "service_id": service_id,
                "status": "resumed",
                "response": response.json()
            }
        else:
            return {
                "error": f"Resume failed with status {response.status_code}",
                "response": response.text
            }
    
    async def _update_action_status(self, action_id: int, status: DeploymentStatus, result: Dict[str, Any]):
        """Update the status and result of a deployment action."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            completed_at = datetime.utcnow() if status in [DeploymentStatus.COMPLETED, DeploymentStatus.FAILED] else None
            
            await conn.execute("""
                UPDATE deployment_actions 
                SET status = $1, result = $2, completed_at = $3
                WHERE id = $4
            """, status.value, result, completed_at, action_id)
        finally:
            await conn.close()
    
    async def get_deployment_action(self, action_id: int) -> Optional[DeploymentAction]:
        """Get a deployment action by ID."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            result = await conn.fetchrow("""
                SELECT id, user_id, action_type, target_service, parameters, status, result, started_at, completed_at
                FROM deployment_actions
                WHERE id = $1
            """, action_id)
            
            if not result:
                return None
            
            return DeploymentAction(
                id=result["id"],
                user_id=result["user_id"],
                action_type=ActionType(result["action_type"]),
                target_service=result["target_service"],
                parameters=result["parameters"],
                status=DeploymentStatus(result["status"]),
                result=result["result"],
                started_at=result["started_at"],
                completed_at=result["completed_at"]
            )
        finally:
            await conn.close()
    
    async def list_deployment_actions(self, user_id: Optional[int] = None, limit: int = 50) -> List[DeploymentAction]:
        """List deployment actions, optionally filtered by user."""
        conn = await asyncpg.connect(self.postgres_dsn)
        try:
            if user_id:
                results = await conn.fetch("""
                    SELECT id, user_id, action_type, target_service, parameters, status, result, started_at, completed_at
                    FROM deployment_actions
                    WHERE user_id = $1
                    ORDER BY started_at DESC
                    LIMIT $2
                """, user_id, limit)
            else:
                results = await conn.fetch("""
                    SELECT id, user_id, action_type, target_service, parameters, status, result, started_at, completed_at
                    FROM deployment_actions
                    ORDER BY started_at DESC
                    LIMIT $1
                """, limit)
            
            return [
                DeploymentAction(
                    id=result["id"],
                    user_id=result["user_id"],
                    action_type=ActionType(result["action_type"]),
                    target_service=result["target_service"],
                    parameters=result["parameters"],
                    status=DeploymentStatus(result["status"]),
                    result=result["result"],
                    started_at=result["started_at"],
                    completed_at=result["completed_at"]
                ) for result in results
            ]
        finally:
            await conn.close()
    
    async def get_available_services(self) -> List[Dict[str, Any]]:
        """Get list of available Render services."""
        if not self.render_api_key:
            return []
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {"Authorization": f"Bearer {self.render_api_key}"}
                
                response = await client.get(
                    f"{self.render_base_url}/services",
                    headers=headers
                )
                
                if response.status_code == 200:
                    services_data = response.json()
                    return [
                        {
                            "id": service["id"],
                            "name": service["name"],
                            "type": service["type"],
                            "status": service["status"],
                            "created_at": service["createdAt"],
                            "suspended": service.get("suspended", "not_suspended")
                        }
                        for service in services_data.get("services", [])
                    ]
                else:
                    return []
        except Exception:
            return []

# Global service instance
_deployment_service: Optional[DeploymentService] = None

def get_deployment_service() -> DeploymentService:
    """Dependency injection for deployment service."""
    global _deployment_service
    if _deployment_service is None:
        postgres_dsn = os.environ.get("RIS_POSTGRES_DSN")
        render_api_key = os.environ.get("RENDER_API_KEY")
        
        if not postgres_dsn:
            raise RuntimeError("RIS_POSTGRES_DSN environment variable not set")
        
        _deployment_service = DeploymentService(postgres_dsn, render_api_key)
    return _deployment_service