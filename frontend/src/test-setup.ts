import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Mock fetch globally
vi.stubGlobal('fetch', vi.fn())

// Mock ResizeObserver
vi.stubGlobal('ResizeObserver', vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
})))