import React, { useState } from 'react'
import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  Database,
  Loader2,
  Send,
  ShieldAlert,
  ShieldCheck,
  UserCheck,
} from 'lucide-react'
import { submitTransaction } from '../services/api'
import { useTransactions } from '../context/TransactionContext'
import ActionBadge from '../components/ActionBadge'

const TRANSACTION_TYPES = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']

const INITIAL_FORM = {
  transaction_id: '',
  type: 'TRANSFER',
  amount: '',
  oldbalanceOrg: '',
  oldbalanceDest: '',
  step: '',
  transaction_day: '',
  transaction_hour: '',
}

export default function NewTransactionPage() {
  const { addTransactionRecord } = useTransactions()
  const [form, setForm] = useState(INITIAL_FORM)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const handleChange = (e) => {
    const { name, value } = e.target
    setForm((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)

    const amount = Number(form.amount)
    const oldbalanceOrg = Number(form.oldbalanceOrg)
    const oldbalanceDest = Number(form.oldbalanceDest)

    if (Number.isNaN(amount) || amount < 0) {
      setError('Amount must be a valid non-negative number.')
      return
    }
    if (Number.isNaN(oldbalanceOrg) || oldbalanceOrg < 0) {
      setError('Origin balance must be a valid non-negative number.')
      return
    }
    if (Number.isNaN(oldbalanceDest) || oldbalanceDest < 0) {
      setError('Destination balance must be a valid non-negative number.')
      return
    }

    const payload = {
      type: form.type,
      amount,
      oldbalanceOrg,
      oldbalanceDest,
    }

    if (form.transaction_id.trim()) {
      payload.transaction_id = form.transaction_id.trim()
    }
    if (form.step !== '') {
      const stepNum = Number(form.step)
      if (!Number.isInteger(stepNum) || stepNum < 1) {
        setError('Step must be an integer >= 1.')
        return
      }
      payload.step = stepNum
    }
    if (form.transaction_day !== '') {
      const dayNum = Number(form.transaction_day)
      if (!Number.isInteger(dayNum) || dayNum < 0 || dayNum > 30) {
        setError('Transaction Day must be an integer between 0 and 30.')
        return
      }
      payload.transaction_day = dayNum
    }
    if (form.transaction_hour !== '') {
      const hourNum = Number(form.transaction_hour)
      if (!Number.isInteger(hourNum) || hourNum < 0 || hourNum > 23) {
        setError('Transaction Hour must be an integer between 0 and 23.')
        return
      }
      payload.transaction_hour = hourNum
    }

    setLoading(true)
    try {
      const data = await submitTransaction(payload)
      setResult(data)
      addTransactionRecord(data)
    } catch (err) {
      if (err.response) {
        const apiErr = err.response.data
        const detailMsg = apiErr?.details
          ? Object.entries(apiErr.details)
              .map(([k, v]) => `${k}: ${v}`)
              .join('; ')
          : apiErr?.message || apiErr?.error || `HTTP ${err.response.status}`
        setError(`Backend Error (${err.response.status}): ${detailMsg}`)
      } else if (err.request) {
        setError(
          'Unable to reach Spring Boot backend. Ensure the backend service is running and CORS/network is accessible.'
        )
      } else {
        setError(err.message || 'Unexpected error submitting transaction.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white">New Transaction Evaluation</h2>
        <p className="text-sm text-slate-400">
          Submit transaction attributes to Spring Boot (<code className="text-cyan-300">POST /api/v1/transactions</code>) for real-time XGBoost scoring, risk policy evaluation, and PostgreSQL persistence.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Form Panel */}
        <form
          onSubmit={handleSubmit}
          className="rounded-xl border border-slate-800/90 bg-navy-900 p-6 lg:col-span-6 space-y-5"
        >
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <label className="block text-xs font-medium uppercase tracking-wider text-slate-300">
                Transaction Type *
              </label>
              <select
                name="type"
                value={form.type}
                onChange={handleChange}
                className="mt-1.5 w-full rounded-lg border border-slate-700 bg-navy-950 px-3 py-2.5 text-sm text-white focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              >
                {TRANSACTION_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium uppercase tracking-wider text-slate-300">
                Transaction ID (Optional)
              </label>
              <input
                type="text"
                name="transaction_id"
                value={form.transaction_id}
                onChange={handleChange}
                placeholder="e.g. tx_987654321"
                className="mt-1.5 w-full rounded-lg border border-slate-700 bg-navy-950 px-3 py-2.5 text-sm text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div>
              <label className="block text-xs font-medium uppercase tracking-wider text-slate-300">
                Amount *
              </label>
              <input
                type="number"
                step="any"
                min="0"
                required
                name="amount"
                value={form.amount}
                onChange={handleChange}
                placeholder="250000.00"
                className="mt-1.5 w-full rounded-lg border border-slate-700 bg-navy-950 px-3 py-2.5 text-sm text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              />
            </div>

            <div>
              <label className="block text-xs font-medium uppercase tracking-wider text-slate-300">
                Origin Balance *
              </label>
              <input
                type="number"
                step="any"
                min="0"
                required
                name="oldbalanceOrg"
                value={form.oldbalanceOrg}
                onChange={handleChange}
                placeholder="260000.00"
                className="mt-1.5 w-full rounded-lg border border-slate-700 bg-navy-950 px-3 py-2.5 text-sm text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              />
            </div>

            <div>
              <label className="block text-xs font-medium uppercase tracking-wider text-slate-300">
                Dest Balance *
              </label>
              <input
                type="number"
                step="any"
                min="0"
                required
                name="oldbalanceDest"
                value={form.oldbalanceDest}
                onChange={handleChange}
                placeholder="0.00"
                className="mt-1.5 w-full rounded-lg border border-slate-700 bg-navy-950 px-3 py-2.5 text-sm text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              />
            </div>
          </div>

          <div className="border-t border-slate-800/80 pt-4">
            <p className="mb-3 text-xs font-medium text-slate-400">
              Temporal Parameters (Optional — day/hour derive from step when omitted)
            </p>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div>
                <label className="block text-xs font-medium uppercase tracking-wider text-slate-300">
                  Step (&ge; 1)
                </label>
                <input
                  type="number"
                  min="1"
                  name="step"
                  value={form.step}
                  onChange={handleChange}
                  placeholder="545"
                  className="mt-1.5 w-full rounded-lg border border-slate-700 bg-navy-950 px-3 py-2.5 text-sm text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium uppercase tracking-wider text-slate-300">
                  Day (0–30)
                </label>
                <input
                  type="number"
                  min="0"
                  max="30"
                  name="transaction_day"
                  value={form.transaction_day}
                  onChange={handleChange}
                  placeholder="22"
                  className="mt-1.5 w-full rounded-lg border border-slate-700 bg-navy-950 px-3 py-2.5 text-sm text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium uppercase tracking-wider text-slate-300">
                  Hour (0–23)
                </label>
                <input
                  type="number"
                  min="0"
                  max="23"
                  name="transaction_hour"
                  value={form.transaction_hour}
                  onChange={handleChange}
                  placeholder="16"
                  className="mt-1.5 w-full rounded-lg border border-slate-700 bg-navy-950 px-3 py-2.5 text-sm text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
                />
              </div>
            </div>
          </div>

          {error && (
            <div className="flex items-start gap-2.5 rounded-lg border border-rose-500/40 bg-rose-500/10 p-3.5 text-sm text-rose-300">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-cyan-500 px-5 py-3 text-sm font-semibold text-navy-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Evaluating Transaction...
              </>
            ) : (
              <>
                <Send className="h-4 w-4" />
                Submit &amp; Score Transaction
              </>
            )}
          </button>
        </form>

        {/* Result Panel */}
        <div className="lg:col-span-6">
          {!result && !loading && (
            <div className="flex h-full min-h-[340px] flex-col items-center justify-center rounded-xl border border-dashed border-slate-700/80 bg-navy-900/50 p-8 text-center">
              <ShieldCheck className="h-10 w-10 text-slate-500" />
              <h3 className="mt-3 text-base font-semibold text-slate-200">
                Awaiting Transaction Submission
              </h3>
              <p className="mt-1.5 max-w-sm text-xs text-slate-400">
                Fill in the transaction fields and submit to inspect the real-time XGBoost
                fraud probability, model threshold comparison, and risk policy decision.
              </p>
            </div>
          )}

          {loading && (
            <div className="flex h-full min-h-[340px] flex-col items-center justify-center rounded-xl border border-slate-800 bg-navy-900 p-8 text-center">
              <Loader2 className="h-9 w-9 animate-spin text-cyan-400" />
              <p className="mt-3 text-sm font-medium text-slate-200">
                Running Spring Boot → FastAPI XGBoost Inference...
              </p>
            </div>
          )}

          {result && !loading && (
            <div className="rounded-xl border border-slate-800/90 bg-navy-900 p-6 space-y-5">
              <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                <div>
                  <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
                    Policy Decision
                  </p>
                  <div className="mt-1.5">
                    <ActionBadge action={result.action} />
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
                    Fraud Probability
                  </p>
                  <p className="mt-1 font-mono text-2xl font-bold text-white">
                    {(result.fraud_probability * 100).toFixed(4)}%
                  </p>
                </div>
              </div>

              {/* Probability vs Threshold Bar */}
              <div>
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Score: {result.fraud_probability.toFixed(6)}</span>
                  <span>Model Threshold: {result.model_threshold.toFixed(6)}</span>
                </div>
                <div className="mt-2 h-2.5 w-full overflow-hidden rounded-full bg-navy-950">
                  <div
                    className={`h-full rounded-full transition-all ${
                      result.action === 'BLOCK'
                        ? 'bg-rose-500'
                        : result.action === 'CHALLENGE'
                        ? 'bg-amber-400'
                        : 'bg-emerald-500'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(2, result.fraud_probability * 100))}%` }}
                  />
                </div>
                <p className="mt-1.5 text-xs text-slate-400">
                  Above ML Threshold:{' '}
                  <span className="font-semibold text-slate-200">
                    {result.is_above_model_threshold ? 'Yes' : 'No'}
                  </span>
                </p>
              </div>

              {/* Governance Controls */}
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="rounded-lg border border-slate-800 bg-navy-950/70 p-3.5">
                  <div className="flex items-center gap-2 text-xs text-slate-400">
                    <ShieldAlert className="h-4 w-4 text-amber-400" />
                    Step-Up MFA Required
                  </div>
                  <p className="mt-1 text-sm font-semibold text-white">
                    {result.step_up_mfa_required ? 'Yes — Challenge User' : 'No'}
                  </p>
                </div>

                <div className="rounded-lg border border-slate-800 bg-navy-950/70 p-3.5">
                  <div className="flex items-center gap-2 text-xs text-slate-400">
                    <UserCheck className="h-4 w-4 text-cyan-400" />
                    Manual Review Required
                  </div>
                  <p className="mt-1 text-sm font-semibold text-white">
                    {result.manual_review_required ? 'Yes — Analyst Queue' : 'No'}
                  </p>
                </div>
              </div>

              {/* Recommendation */}
              <div className="rounded-lg border border-slate-800 bg-navy-950/70 p-4">
                <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
                  Policy Recommendation
                </p>
                <p className="mt-1 text-sm text-slate-200">{result.recommendation}</p>
              </div>

              {/* Metadata */}
              <div className="grid grid-cols-2 gap-3 border-t border-slate-800 pt-4 text-xs text-slate-400">
                <div className="flex items-center gap-1.5">
                  <Cpu className="h-3.5 w-3.5 text-cyan-400" />
                  <span>Model: {result.model_version}</span>
                </div>
                <div className="text-right font-mono">
                  Latency: {result.ml_latency_ms} ms
                </div>
                <div className="col-span-2 flex items-center gap-1.5 font-mono text-[11px] text-slate-500">
                  <Database className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                  <span className="truncate">Persisted ID: {result.id}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
