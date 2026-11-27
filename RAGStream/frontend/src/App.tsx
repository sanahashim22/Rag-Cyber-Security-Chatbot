import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from './api/client'
import { Header } from './components/Header'
import { Sidebar } from './components/Sidebar'
import { AlertInputsStatus } from './components/AlertInputsStatus'
import { ReceiveFromIdsDashboard } from './components/ReceiveFromIdsDashboard'
import { IntelDbPanel } from './components/IntelDbPanel'
import { ALL_FILES_CHOICE, AutoRunLimitConfirm, FileDropdown } from './components/FileSelector'
import { Tabs, Callout } from './components/Common'
import { SingleModelPanel } from './components/SingleModelPanel'
import { ComparisonPanel } from './components/ComparisonPanel'
import { ChatWidget } from './components/ChatWidget'

export default function App() {
  const { data: meta } = useQuery({ queryKey: ['meta'], queryFn: api.meta })
  const { data: modelsInfo } = useQuery({ queryKey: ['models'], queryFn: api.models })
  const { data: scan } = useQuery({ queryKey: ['alerts-scan'], queryFn: api.scanAlerts, refetchInterval: 4000 })

  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [idsBaseUrl, setIdsBaseUrl] = useState('http://172.16.20.142:8501')
  const [idsToken, setIdsToken] = useState('')
  const [fileChoice, setFileChoice] = useState(ALL_FILES_CHOICE)
  const [confirmedCount, setConfirmedCount] = useState<number | null>(null)
  const [mainTab, setMainTab] = useState('single')

  const [singleBusy, setSingleBusy] = useState(false)
  const [compareBusy, setCompareBusy] = useState(false)
  const anyBusy = singleBusy || compareBusy

  const fhash = `${scan?.fhash ?? ''}::${fileChoice}`
  useEffect(() => {
    setConfirmedCount(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fhash])

  const { data: entriesResult } = useQuery({
    queryKey: ['alert-entries', fileChoice],
    queryFn: () => api.getEntries(fileChoice === ALL_FILES_CHOICE ? undefined : fileChoice),
    enabled: !!scan && scan.file_summaries.length > 0,
  })

  const [chatModel, setChatModel] = useState('llama3.2')

  if (!scan || !modelsInfo) {
    return (
      <div className="app-shell">
        <div className="app-sidebar" />
        <div className="app-main">
          <Header />
          <Callout kind="info">Loading…</Callout>
        </div>
      </div>
    )
  }

  const sidebarToggle = (
    <button
      className="sidebar-toggle"
      onClick={() => setSidebarOpen((v) => !v)}
      title={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
      type="button"
    >
      {sidebarOpen ? '◀' : '▶'}
    </button>
  )

  const autoRunLimit = modelsInfo.auto_run_limit
  const totalInFile =
    fileChoice === ALL_FILES_CHOICE
      ? scan.total_parsed
      : scan.file_summaries.find((f) => f.name === fileChoice)?.count ?? 0

  let rankableEntries = entriesResult?.entries ?? []
  const needsConfirm = totalInFile > autoRunLimit && confirmedCount === null
  if (!needsConfirm && confirmedCount != null) {
    rankableEntries = rankableEntries.slice(0, confirmedCount)
  }

  return (
    <div className="app-shell">
      <div className={`app-sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <Sidebar meta={meta} />
      </div>
      {sidebarToggle}
      <div className="app-main">
        <Header />

        <AlertInputsStatus />
        <ReceiveFromIdsDashboard baseUrl={idsBaseUrl} token={idsToken} onChange={(b, t) => { setIdsBaseUrl(b); setIdsToken(t) }} />
        <IntelDbPanel baseUrl={idsBaseUrl} token={idsToken} />

        {scan.file_summaries.length === 0 ? (
          <ChatWidget model={chatModel} busy={anyBusy} />
        ) : (
          <>
            <FileDropdown scan={scan} value={fileChoice} onChange={setFileChoice} />

            {totalInFile === 0 ? (
              <>
                <Callout kind="warn">No valid alerts found in `{fileChoice}`.</Callout>
                <ChatWidget model={chatModel} busy={anyBusy} />
              </>
            ) : needsConfirm ? (
              <>
                <AutoRunLimitConfirm
                  fileChoice={fileChoice}
                  totalParsed={totalInFile}
                  autoRunLimit={autoRunLimit}
                  onConfirm={setConfirmedCount}
                />
                <ChatWidget model={chatModel} busy={anyBusy} />
              </>
            ) : !entriesResult ? (
              <Callout kind="info">Loading alerts…</Callout>
            ) : (
              <>
                {confirmedCount != null && confirmedCount < totalInFile && (
                  <p className="small-muted">
                    Processing the first {confirmedCount} of {totalInFile} alerts (as selected above).
                  </p>
                )}

                <Tabs
                  active={mainTab}
                  onChange={setMainTab}
                  tabs={[
                    { key: 'single', label: '🔎 Single-Model Ranking' },
                    { key: 'compare', label: '📊 4-Model Comparison' },
                  ]}
                />

                {/* Both panels stay mounted (just hidden) rather than being
                    swapped in/out - each one's running job is tracked in its
                    own local state, so unmounting it when you switch tabs
                    would silently lose track of a job that's still running
                    on the backend. */}
                <div style={{ display: mainTab === 'single' ? 'block' : 'none' }}>
                  <SingleModelPanel
                    entries={rankableEntries}
                    models={modelsInfo.models}
                    meta={meta}
                    onBusyChange={setSingleBusy}
                    onModelChange={setChatModel}
                  />
                </div>
                <div style={{ display: mainTab === 'compare' ? 'block' : 'none' }}>
                  <ComparisonPanel entries={rankableEntries} models={modelsInfo.comparison_models} onBusyChange={setCompareBusy} />
                </div>

                <ChatWidget model={chatModel} busy={anyBusy} />
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}
