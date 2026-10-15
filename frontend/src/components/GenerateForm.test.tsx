import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import { GenerateForm } from './GenerateForm'
import { api, GenerateResponse } from '../api/client'

// Mock the API client
vi.mock('../api/client', () => ({
  api: {
    generate: vi.fn(),
  },
}))

describe('GenerateForm', () => {
  const mockGenerate = vi.mocked(api.generate)

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders form with default values', () => {
    render(<GenerateForm />)

    expect(screen.getByLabelText('Prompt')).toHaveValue('The capital of France is')
    // Number inputs return number values
    expect(screen.getByLabelText('Max New Tokens')).toHaveValue(32)
    expect(screen.getByLabelText('Temperature')).toHaveValue(0)
    expect(screen.getByLabelText('Top-p')).toHaveValue(1)
    expect(screen.getByLabelText('Top-k')).toHaveValue(0)
    expect(screen.getByLabelText('Repetition Penalty')).toHaveValue(1)
    // Empty number input may return empty string or null
    const seedInput = screen.getByLabelText('Seed (optional)')
    expect(seedInput).toBeInTheDocument()
  })

  it('shows loading state during generation', async () => {
    // Use a delayed promise to catch loading state
    let resolveMock: (value: GenerateResponse) => void
    const promise = new Promise<GenerateResponse>((resolve) => {
      resolveMock = resolve
    })
    mockGenerate.mockReturnValueOnce(promise)

    render(<GenerateForm />)

    const submitButton = screen.getByRole('button', { name: /generate/i })
    await act(async () => {
      fireEvent.click(submitButton)
    })

    // Button should show "Generating..." and be disabled
    const loadingButton = screen.getByRole('button', { name: /generating/i })
    expect(loadingButton).toBeDisabled()

    // Resolve the promise
    await act(async () => {
      resolveMock!({
        completion: 'Paris is the capital.',
        tokens_generated: 5,
        latency_ms: 42.5,
        model_name: 'test-model',
        seed_used: 123,
      })
      await promise
    })
  })

  it('calls onGenerate callback with response', async () => {
    const onGenerate = vi.fn()
    mockGenerate.mockResolvedValueOnce({
      completion: 'Callback test',
      tokens_generated: 2,
      latency_ms: 15,
      model_name: 'test-model',
      seed_used: 999,
    })

    render(<GenerateForm onGenerate={onGenerate} />)

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /generate/i }))
      await waitFor(() => expect(onGenerate).toHaveBeenCalledTimes(1))
    })

    expect(onGenerate).toHaveBeenCalledWith(
      expect.objectContaining({
        completion: 'Callback test',
        tokens_generated: 2,
        latency_ms: 15,
        seed_used: 999,
      })
    )
  })

  it('updates prompt value when typing', async () => {
    render(<GenerateForm />)

    const textarea = screen.getByLabelText('Prompt')
    await act(async () => {
      fireEvent.change(textarea, { target: { value: 'New prompt' } })
    })

    expect(textarea).toHaveValue('New prompt')
  })

  it('disables generate button when prompt is empty', async () => {
    render(<GenerateForm />)

    const textarea = screen.getByLabelText('Prompt')
    const submitButton = screen.getByRole('button', { name: /generate/i })

    await act(async () => {
      fireEvent.change(textarea, { target: { value: '' } })
    })

    expect(submitButton).toBeDisabled()
  })
})