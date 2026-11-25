import { useEffect, useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { api } from '../api/client'

interface ChatMsg {
  role: 'user' | 'assistant'
  content: string
}

export function ChatWidget({ model, busy }: { model: string; busy: boolean }) {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<ChatMsg[]>([])
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)

  const send = useMutation({
    mutationFn: (question: string) => api.chatSend(question, model),
    onSuccess: (res) => setMessages((m) => [...m, { role: 'assistant', content: res.answer }]),
    onError: (e: any) => setMessages((m) => [...m, { role: 'assistant', content: `Sorry, I hit an error answering that: ${e}` }]),
  })

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [messages, send.isPending])

  function submit(e: React.FormEvent) {
    e.preventDefault()
    const q = input.trim()
    if (!q || send.isPending) return
    setMessages((m) => [...m, { role: 'user', content: q }])
    setInput('')
    send.mutate(q)
  }

  if (!open) {
    return (
      <button className="chatbot-fab" disabled={busy} onClick={() => setOpen(true)} title="Ask a cybersecurity question">
        🤖
      </button>
    )
  }

  return (
    <>
      <div className="chatbot-overlay" onClick={() => setOpen(false)} />
      <div className={`chatbot-panel ${busy ? 'busy' : ''}`}>
        <div className="chatbot-header">
          <div className="chatbot-title">
            <span className="chatbot-dot" /> 🤖 Security Assistant
          </div>
          <button className="chatbot-minimize" onClick={() => setOpen(false)} title="Minimize">
            ➖
          </button>
        </div>

        {messages.length === 0 ? (
          <p className="small-muted" style={{ marginTop: 0 }}>
            💬 We're here to answer your security questions using the NIST CSF 2.0 / ISO 27001 standards. Using{' '}
            <b>{model}</b> for responses.
          </p>
        ) : (
          <p className="small-muted" style={{ marginTop: 0 }}>
            Using <b>{model}</b> for responses.
          </p>
        )}

        <div className="chatbot-messages" ref={scrollRef}>
          {messages.map((m, i) => (
            <div key={i} className={`chat-message ${m.role}`}>
              <div className="chat-message-role">{m.role === 'user' ? 'You' : 'Assistant'}</div>
              {m.content}
            </div>
          ))}
          {send.isPending && <div className="chat-message">Thinking…</div>}
        </div>

        <form className="chatbot-form" onSubmit={submit}>
          <input
            type="text"
            placeholder="Ask about NIST CSF, ISO 27001…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={busy}
          />
          <button className="chatbot-send" type="submit" disabled={busy || send.isPending}>
            Send
          </button>
        </form>
      </div>
    </>
  )
}
