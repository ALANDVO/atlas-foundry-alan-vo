import { useState, useCallback } from 'react'
import { api, GenerateRequest, GenerateResponse } from '../api/client'

interface GenerateFormProps {
  onGenerate?: (response: GenerateResponse) => void
}

export function GenerateForm({ onGenerate }: GenerateFormProps) {
  const [prompt, setPrompt] = useState('The capital of France is')
  const [maxNewTokens, setMaxNewTokens] = useState(32)
  const [temperature, setTemperature] = useState(0.0)
  const [topP, setTopP] = useState(1.0)
  const [topK, setTopK] = useState(0)
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.0)
  const [seed, setSeed] = useState<number | ''>('')
  const [completion, setCompletion] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [latency, setLatency] = useState<number | null>(null)
  const [tokensGenerated, setTokensGenerated] = useState<number | null>(null)
  const [seedUsed, setSeedUsed] = useState<number | null>(null)

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    const request: GenerateRequest = {
      prompt,
      max_new_tokens: maxNewTokens,
      temperature,
      top_p: topP,
      top_k: topK,
      repetition_penalty: repetitionPenalty,
      seed: seed !== '' ? Number(seed) : undefined,
    }

    try {
      const response = await api.generate(request)
      setCompletion(response.completion)
      setLatency(response.latency_ms)
      setTokensGenerated(response.tokens_generated)
      setSeedUsed(response.seed_used)
      onGenerate?.(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generation failed')
      setCompletion(null)
    } finally {
      setLoading(false)
    }
  }, [prompt, maxNewTokens, temperature, topP, topK, repetitionPenalty, seed, onGenerate])

  const handleClear = useCallback(() => {
    setCompletion(null)
    setError(null)
    setLatency(null)
    setTokensGenerated(null)
    setSeedUsed(null)
  }, [])

  return (
    <section className="generate-form" style={styles.section}>
      <header style={styles.header}>
        <h2 style={styles.title}>Text Generation</h2>
        <p style={styles.subtitle}>TokenSpeed inference endpoint</p>
      </header>

      <form onSubmit={handleSubmit} style={styles.form}>
        <div style={styles.fieldGroup}>
          <label htmlFor="prompt" style={styles.label}>
            Prompt
          </label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            placeholder="Enter your prompt..."
            style={styles.textarea}
            disabled={loading}
          />
        </div>

        <div style={styles.paramsGrid}>
          <div style={styles.fieldGroup}>
            <label htmlFor="maxNewTokens" style={styles.label}>
              Max New Tokens
            </label>
            <input
              id="maxNewTokens"
              type="number"
              value={maxNewTokens}
              onChange={(e) => setMaxNewTokens(Number(e.target.value))}
              min={1}
              max={2048}
              style={styles.input}
              disabled={loading}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label htmlFor="temperature" style={styles.label}>
              Temperature
            </label>
            <input
              id="temperature"
              type="number"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              min={0}
              max={2}
              step={0.1}
              style={styles.input}
              disabled={loading}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label htmlFor="topP" style={styles.label}>
              Top-p
            </label>
            <input
              id="topP"
              type="number"
              value={topP}
              onChange={(e) => setTopP(Number(e.target.value))}
              min={0}
              max={1}
              step={0.05}
              style={styles.input}
              disabled={loading}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label htmlFor="topK" style={styles.label}>
              Top-k
            </label>
            <input
              id="topK"
              type="number"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              min={0}
              max={100}
              style={styles.input}
              disabled={loading}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label htmlFor="repetitionPenalty" style={styles.label}>
              Repetition Penalty
            </label>
            <input
              id="repetitionPenalty"
              type="number"
              value={repetitionPenalty}
              onChange={(e) => setRepetitionPenalty(Number(e.target.value))}
              min={0.1}
              max={2}
              step={0.1}
              style={styles.input}
              disabled={loading}
            />
          </div>

          <div style={styles.fieldGroup}>
            <label htmlFor="seed" style={styles.label}>
              Seed (optional)
            </label>
            <input
              id="seed"
              type="number"
              value={seed}
              onChange={(e) => setSeed(e.target.value === '' ? '' : Number(e.target.value))}
              min={0}
              max={2147483647}
              placeholder="Random"
              style={styles.input}
              disabled={loading}
            />
          </div>
        </div>

        <div style={styles.buttonGroup}>
          <button
            type="submit"
            disabled={loading || !prompt.trim()}
            style={{ ...styles.button, ...(loading || !prompt.trim() ? styles.buttonDisabled : {}) }}
          >
            {loading ? 'Generating...' : 'Generate'}
          </button>
          <button
            type="button"
            onClick={handleClear}
            disabled={loading}
            style={styles.buttonSecondary}
          >
            Clear
          </button>
        </div>

        {error && (
          <div style={styles.error} role="alert" data-testid="error-message">
            Error: {error}
          </div>
        )}
      </form>

      {completion !== null && (
        <section style={styles.outputSection}>
          <header style={styles.outputHeader}>
            <h3 style={styles.outputTitle}>Completion</h3>
            <div style={styles.meta}>
              {latency !== null && <span>⏱ {latency.toFixed(1)} ms</span>}
              {tokensGenerated !== null && <span>🔤 {tokensGenerated} tokens</span>}
              {seedUsed !== null && <span>🎲 seed: {seedUsed}</span>}
            </div>
          </header>
          <pre style={styles.output} data-testid="completion-output">{completion}</pre>
        </section>
      )}
    </section>
  )
}

const styles: Record<string, React.CSSProperties> = {
  section: {
    backgroundColor: 'var(--color-bg-secondary)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-lg)',
    padding: 'var(--spacing-lg)',
    maxWidth: '800px',
    margin: '0 auto',
  },
  header: {
    marginBottom: 'var(--spacing-lg)',
  },
  title: {
    fontSize: '1.5rem',
    fontWeight: 600,
    color: 'var(--color-text-primary)',
    marginBottom: 'var(--spacing-xs)',
  },
  subtitle: {
    fontSize: '0.875rem',
    color: 'var(--color-text-secondary)',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: 'var(--spacing-lg)',
  },
  fieldGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: 'var(--spacing-xs)',
  },
  label: {
    fontSize: '0.875rem',
    fontWeight: 500,
    color: 'var(--color-text-primary)',
  },
  textarea: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.9375rem',
    lineHeight: 1.6,
    resize: 'vertical',
    minHeight: '100px',
  },
  paramsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
    gap: 'var(--spacing-md)',
  },
  input: {
    fontSize: '0.9375rem',
  },
  buttonGroup: {
    display: 'flex',
    gap: 'var(--spacing-md)',
    marginTop: 'var(--spacing-sm)',
  },
  button: {
    backgroundColor: 'var(--color-accent)',
    color: 'white',
    fontWeight: 600,
    padding: 'var(--spacing-sm) var(--spacing-lg)',
    borderRadius: 'var(--radius-md)',
    fontSize: '1rem',
    transition: 'background-color var(--transition-fast), transform var(--transition-fast)',
  },
  buttonDisabled: {
    backgroundColor: 'var(--color-border)',
    color: 'var(--color-text-secondary)',
    cursor: 'not-allowed',
  },
  buttonSecondary: {
    backgroundColor: 'transparent',
    color: 'var(--color-text-secondary)',
    fontWeight: 500,
    padding: 'var(--spacing-sm) var(--spacing-lg)',
    borderRadius: 'var(--radius-md)',
    border: '1px solid var(--color-border)',
    transition: 'border-color var(--transition-fast), color var(--transition-fast)',
  },
  error: {
    backgroundColor: 'rgba(248, 81, 73, 0.1)',
    border: '1px solid var(--color-error)',
    color: 'var(--color-error)',
    padding: 'var(--spacing-md)',
    borderRadius: 'var(--radius-md)',
    fontSize: '0.875rem',
  },
  outputSection: {
    marginTop: 'var(--spacing-xl)',
    paddingTop: 'var(--spacing-lg)',
    borderTop: '1px solid var(--color-border)',
  },
  outputHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 'var(--spacing-md)',
    flexWrap: 'wrap',
    gap: 'var(--spacing-sm)',
  },
  outputTitle: {
    fontSize: '1.125rem',
    fontWeight: 600,
  },
  meta: {
    display: 'flex',
    gap: 'var(--spacing-md)',
    fontSize: '0.8125rem',
    color: 'var(--color-text-secondary)',
    fontFamily: 'var(--font-mono)',
  },
  output: {
    backgroundColor: 'var(--color-bg)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-md)',
    padding: 'var(--spacing-md)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.9375rem',
    lineHeight: 1.7,
    whiteSpace: 'pre-wrap',
    wordWrap: 'break-word',
    overflowX: 'auto',
    maxHeight: '400px',
    overflowY: 'auto',
  },
}