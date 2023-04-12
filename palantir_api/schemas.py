
from pydantic import BaseModel


class Config(BaseModel):
    clientId: str
    tenantId: str
    azureOAuthEnable: bool
