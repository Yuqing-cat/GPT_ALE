import { PublicClientApplication } from '@azure/msal-browser'
import type { Configuration } from '@azure/msal-browser'

const getMsalInstance = (config: any) => {
  const msalConfig: Configuration = {
    auth: {
      clientId: config.AZURE_CLIENT_ID,
      authority: `https://login.microsoftonline.com/${config.AZURE_TENANT_ID}`,
      redirectUri: window.location.origin
    }
  }
  return new PublicClientApplication(msalConfig)
}

const config: any = {
  API_PATH: process.env.API_PATH,
  AZURE_ENABLE: false,
  AZURE_TENANT_ID: '',
  AZURE_CLIENT_ID: '',
  VERSION: process.env.VERSION,
  GENERATED_TIME: process.env.GENERATED_TIME,
  msalInstance: null
}

export const initAzureConfig = (data: any) => {
  if (data.azureOAuthEnable && data.clientId && data.tenantId) {
    config.AZURE_ENABLE = data.azureOAuthEnable
    config.AZURE_CLIENT_ID = data.clientId
    config.AZURE_TENANT_ID = data.tenantId
    config.msalInstance = getMsalInstance(config)
  }
}

export default config
