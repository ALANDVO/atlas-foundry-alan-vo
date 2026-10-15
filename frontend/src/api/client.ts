/**
 * Typed API Client for Atlas Foundry (TokenSpeed) Backend.
 *
 * Derived from backend OpenAPI specification.
 * Update this when backend routes change.
 */

const API_BASE = '/v1'

// Request/Response Types
export interface GenerateRequest {
  prompt: string
  max_new_tokens?: number
  temperature?: number
  top_p?: number
  top_k?: number
  repetition_penalty?: number
  seed?: number
}

export interface GenerateResponse {
  completion: string
  tokens_generated: number
  latency_ms: number
  model_name: string
  seed_used: number | null
}

export interface HealthResponse {
  status: 'healthy' | 'initializing' | 'degraded'
  version: string
  model_loaded: boolean
  model_name: string
}

export interface ModelInfoResponse {
  model_name: string
  loaded: boolean
  dtype: string
  tensor_parallel_size: number
}

export interface RootResponse {
  name: string
  version: string
  description: string
  author: string
  docs: string
}

export interface ApiError {
  detail: string
}

// Generic fetch wrapper with error handling
async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  })

  if (!response.ok) {
    let detail = `HTTP ${response.status}: ${response.statusText}`
    try {
      const errorData = await response.json()
      detail = errorData.detail || detail
    } catch {
      // Use default detail
    }
    throw new Error(detail)
  }

  if (response.status === 204) {
    return undefined as T
  }

  return response.json()
}

// API Methods
export const api = {
  // Generation
  generate: (request: GenerateRequest): Promise<GenerateResponse> =>
    fetchJson<GenerateResponse>('/generate', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  // Health & Info
  health: (): Promise<HealthResponse> =>
    fetchJson<HealthResponse>('/health'),

  modelInfo: (): Promise<ModelInfoResponse> =>
    fetchJson<ModelInfoResponse>('/model/info'),

  root: (): Promise<RootResponse> =>
    fetchJson<RootResponse>('/'),
}

// React Query-style hooks (simplified, no external deps)
export function useApi() {
  return api
}

// Helper for deterministic seed generation
export function generateSeed(): number {
  return Math.floor(Math.random() * 2_147_483_647)
}