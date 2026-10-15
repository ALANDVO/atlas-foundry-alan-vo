import { GenerateForm } from './components/GenerateForm'
import { api } from './api/client'
import { useState, useEffect } from 'react'

function App() {
  const [health, setHealth] = useState<{ status: string; model_loaded: boolean } | null>(null)

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const h = await api.health()
        setHealth(h)
      } catch {
        setHealth({ status: 'error', model_loaded: false })
      }
    }
    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div style={styles.logo}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
            <span style={styles.logoText}>Atlas Foundry</span>
          </div>
          <nav style={styles.nav}>
            <span style={styles.statusBadge}>
              {health?.model_loaded ? (
                <>
                  <span style={{ ...styles.statusDot, backgroundColor: 'var(--color-success)' }} aria-hidden="true" />
                  Ready
                </>
              ) : (
                <>
                  <span style={{ ...styles.statusDot, backgroundColor: 'var(--color-warning)' }} aria-hidden="true" />
                  Initializing...
                </>
              )}
            </span>
          </nav>
        </div>
      </header>

      <main style={styles.main}>
        <div style={styles.container}>
          <GenerateForm />
        </div>
      </main>

      <footer style={styles.footer}>
        <p>Atlas Foundry (TokenSpeed) — Speed-of-light LLM inference engine</p>
        <p style={styles.footerLinks}>
          <a href="https://github.com/ALANDVO/atlas-foundry-alan-vo" target="_blank" rel="noopener noreferrer">GitHub</a>
          <span aria-hidden="true">·</span>
          <a href="/docs" target="_blank" rel="noopener noreferrer">API Docs</a>
          <span aria-hidden="true">·</span>
          {'Alan Vo <alanvo@gmail.com>'}
        </p>
      </footer>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  app: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    backgroundColor: 'var(--color-bg-secondary)',
    borderBottom: '1px solid var(--color-border)',
    padding: 'var(--spacing-md) var(--spacing-xl)',
    position: 'sticky',
    top: 0,
    zIndex: 100,
  },
  headerContent: {
    maxWidth: '1200px',
    margin: '0 auto',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-sm)',
  },
  logoText: {
    fontSize: '1.25rem',
    fontWeight: 700,
    color: 'var(--color-text-primary)',
    letterSpacing: '-0.02em',
  },
  nav: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-md)',
  },
  statusBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-xs)',
    fontSize: '0.8125rem',
    fontWeight: 500,
    color: 'var(--color-text-secondary)',
  },
  statusDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
  },
  main: {
    flex: 1,
    padding: 'var(--spacing-xl) var(--spacing-md)',
    width: '100%',
  },
  container: {
    maxWidth: '900px',
    margin: '0 auto',
    width: '100%',
  },
  footer: {
    backgroundColor: 'var(--color-bg-secondary)',
    borderTop: '1px solid var(--color-border)',
    padding: 'var(--spacing-lg) var(--spacing-xl)',
    textAlign: 'center',
  },
  footerLinks: {
    marginTop: 'var(--spacing-xs)',
    fontSize: '0.8125rem',
    color: 'var(--color-text-secondary)',
  },
}

export default App