import React, { useMemo, useState } from 'react'
import {
  AlertCircle,
  ArrowRightLeft,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Cpu,
  Database,
  KeyRound,
  Loader2,
  Send,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  UserCheck,
  XOctagon,
} from 'lucide-react'
import { submitTransaction } from '../services/api'
import { useTransactions } from '../context/TransactionContext'

const TRANSACTION_TYPES = [
  { value: 'TRANSFER', label: 'Bank Transfer' },
  { value: 'PAYMENT', label: 'Merchant Payment' },
  { value: 'CASH_OUT', label: 'Cash Out / Withdrawal' },
  { value: 'CASH_IN', label: 'Cash In / Deposit' },
  { value: 'DEBIT', label: 'Direct Debit' },
]

// Curated demo accounts so ordinary users can test realistic balances without manual ML feature entry.
const DEMO_ACCOUNTS = [
  { id: 'ACC-1001', label: 'Alex Rivera — Checking', balance: 150000.0 },
  { id: 'ACC-1002', label: 'Sarah Chen — Savings', balance: 42500.0 },
  { id: 'ACC-2045', label: 'CloudHost Merchant', balance: 8200.0 },
  { id: 'ACC-8890', label: 'External / New Account', balance: 0.0 },
]

function generateTransactionId() {
  const randomPart = Math.random().toString(36).substring(2, 8).toUpperCase()
  return `TXN-${Date.now().toString().slice(-6)}-${randomPart}`
}

function deriveTemporalFeatures(date = new Date()) {
  const day = Math.min(30, Math.max(0, date.getDate() - 1)) // backend expects 0..30
  const hour = date.getHours() // 0..23
  const step = day * 24 + hour + 1 // >= 1
  return { step, transaction_day: day, transaction_hour: hour }
}

function getDecisionMeta(result) {
  if (!result) return null
  const { action, manual_review_required: manualReview, fraud_probability: prob } = result

  let decisionLabel = 'Declined'
  let decisionTone = 'rose'
  let DecisionIcon = XOctagon

  if (action === 'APPROVE') {
    decisionLabel = 'Approved'
    decisionTone = 'emerald'
    DecisionIcon = CheckCircle2
  } else if (action === 'CHALLENGE' || manualReview) {
    decisionLabel = 'Review'
    decisionTone = 'amber'
    DecisionIcon = ShieldAlert
  }

  let riskLevel = 'Low'
  let riskTone = 'emerald'
  if (prob >= 0.7 || action === 'BLOCK') {
    riskLevel = 'High'
    riskTone = 'rose'
  } else if (prob >= 0.3 || action === 'CHALLENGE' || manualReview) {
    riskLevel = 'Medium'
    riskTone = 'amber'
  }

  return { decisionLabel, decisionTone, DecisionIcon, riskLevel, riskTone }
}

const TONE_CLASSES = {
  emerald: {
    badge: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300',
    card: 'border-emerald-500/30 bg-emerald-500/5',
    bar: 'bg-emerald-400',
    text: 'text-emerald-400',
  },
  amber: {
    badge: 'border-amber-500/30 bg-amber-500/10 text-amber-300',
    card: 'border-amber-500/30 bg-amber-500/5',
    bar: 'bg-amber-400',
    text: 'text-amber-400',
  },
  rose: {
    badge: 'border-rose-500/30 bg-rose-500/10 text-rose-300',
    card: 'border-rose-500/30 bg-rose-500/5',
    bar: 'bg-rose-400',
    text: 'text-rose-400',
  },
}

export default function NewTransactionPage() {
  const { addTransactionRecord } = useTransactions()

  const [type, setType] = useState('TRANSFER')
  const [amount, setAmount] = useState('')
  const [senderAccount, setSenderAccount] = useState(DEMO_ACCOUNTS[0].id)
  const [receiverAccount, setReceiverAccount] = useState(DEMO_ACCOUNTS[3].id)

  // Advanced / ML feature overrides
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [customOriginBalance, setCustomOriginBalance] = useState('')
  const [customDestBalance, setCustomDestBalance] = useState('')
  const [customTxId, setCustomTxId] = useState('')
  const [customStep, setCustomStep] = useState('')
  const [customDay, setCustomDay] = useState('')
  const [customHour, setCustomHour] = useState('')

  // Result & UI states
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false)

  const senderDemo = useMemo(
    () => DEMO_ACCOUNTS.find((a) => a.id === senderAccount) || null,
    [senderAccount]
  )
  const receiverDemo = useMemo(
    () => DEMO_ACCOUNTS.find((a) => a.id === receiverAccount) || null,
    [receiverAccount]
  )

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)

    const parsedAmount = Number(amount)
    if (!amount || Number.isNaN(parsedAmount) || parsedAmount <= 0) {
      setError('Please enter a valid transaction amount greater than 0.')
      return
    }

    if (!senderAccount.trim() || !receiverAccount.trim()) {
      setError('Please select or enter both Sender and Receiver accounts.')
      return
    }

    // Resolve balances: use explicit override if provided, otherwise demo account balance.
    // Never invent a fake balance for an unknown custom account.
    let oldbalanceOrg =
      customOriginBalance !== '' ? Number(customOriginBalance) : senderDemo?.balance
    let oldbalanceDest =
      customDestBalance !== '' ? Number(customDestBalance) : receiverDemo?.balance

    if (oldbalanceOrg === undefined || Number.isNaN(oldbalanceOrg) || oldbalanceOrg < 0) {
      setShowAdvanced(true)
      setError(
        'Sender account balance is unavailable. Please select a demo account or enter Sender Balance in Advanced Transaction Details.'
      )
      return
    }

    if (oldbalanceDest === undefined || Number.isNaN(oldbalanceDest) || oldbalanceDest < 0) {
      setShowAdvanced(true)
      setError(
        'Receiver account balance is unavailable. Please select a demo account or enter Receiver Balance in Advanced Transaction Details.'
      )
      return
    }

    const temporal = deriveTemporalFeatures()
    const payload = {
      transaction_id: customTxId.trim() || generateTransactionId(),
      type,
      amount: parsedAmount,
      oldbalanceOrg,
      oldbalanceDest,
      step: customStep !== '' ? Number(customStep) : temporal.step,
      transaction_day: customDay !== '' ? Number(customDay) : temporal.transaction_day,
      transaction_hour: customHour !== '' ? Number(customHour) : temporal.transaction_hour,
    }

    if (!Number.isInteger(payload.step) || payload.step < 1) {
      setError('Step must be an integer >= 1.')
      return
    }
    if (
      !Number.isInteger(payload.transaction_day) ||
      payload.transaction_day < 0 ||
      payload.transaction_day > 30
    ) {
      setError('Transaction Day must be an integer between 0 and 30.')
      return
    }
    if (
      !Number.isInteger(payload.transaction_hour) ||
      payload.transaction_hour < 0 ||
      payload.transaction_hour > 23
    ) {
      setError('Transaction Hour must be an integer between 0 and 23.')
      return
    }

    setLoading(true)
    try {
      const data = await submitTransaction(payload)
      setResult(data)
      setShowTechnicalDetails(false)
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
          'Unable to reach the SentinelPay backend. Please ensure the Spring Boot service is running.'
        )
      } else {
        setError(err.message || 'Unexpected error submitting transaction.')
      }
    } finally {
      setLoading(false)
    }
  }

  const decisionMeta = useMemo(() => getDecisionMeta(result), [result])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white">
            New Transaction
          </h2>
          <p className="text-sm text-slate-400">
            Submit a payment for real-time AI fraud screening and risk assessment.
          </p>
        </div>
        <div className="inline-flex items-center gap-2 self-start rounded-full border border-cyan-500/25 bg-cyan-500/10 px-3 py-1 text-xs font-medium text-cyan-300">
          <Sparkles className="h-3.5 w-3.5" />
          Real-Time Fraud Engine Active
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
        {/* Simplified Transaction Form */}
        <form
          onSubmit={handleSubmit}
          className="rounded-2xl border border-slate-800/90 bg-navy-900/95 p-6 shadow-xl lg:col-span-6 space-y-5"
        >
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-cyan-500/15 text-cyan-400">
                <ArrowRightLeft className="h-4 w-4" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-white">Payment Details</h3>
                <p className="text-xs text-slate-400">
                  Transaction ID &amp; timestamps are generated automatically
                </p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {/* 1. Transaction Type */}
            <div>
              <label
                htmlFor="tx-type"
                className="block text-xs font-medium uppercase tracking-wider text-slate-300"
              >
                Transaction Type
              </label>
              <select
                id="tx-type"
                value={type}
                onChange={(e) => setType(e.target.value)}
                className="mt-1.5 w-full rounded-xl border border-slate-700/80 bg-navy-950 px-3.5 py-2.5 text-sm text-white transition focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              >
                {TRANSACTION_TYPES.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label}
                  </option>
                ))}
              </select>
            </div>

            {/* 2. Transaction Amount */}
            <div>
              <label
                htmlFor="tx-amount"
                className="block text-xs font-medium uppercase tracking-wider text-slate-300"
              >
                Transaction Amount (USD)
              </label>
              <div className="relative mt-1.5">
                <span className="pointer-events-none absolute inset-y-0 left-3.5 flex items-center text-sm font-medium text-slate-400">
                  $
                </span>
                <input
                  id="tx-amount"
                  type="number"
                  step="0.01"
                  min="0.01"
                  required
                  value={amount}
                  onChange={(e) => setAmount(e.target.value)}
                  placeholder="0.00"
                  className="w-full rounded-xl border border-slate-700/80 bg-navy-950 py-2.5 pl-8 pr-3.5 text-sm text-white placeholder-slate-500 transition focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
                />
              </div>
            </div>

            {/* 3. Sender Account */}
            <div>
              <label
                htmlFor="tx-sender"
                className="block text-xs font-medium uppercase tracking-wider text-slate-300"
              >
                Sender Account
              </label>
              <select
                id="tx-sender"
                value={senderAccount}
                onChange={(e) => setSenderAccount(e.target.value)}
                className="mt-1.5 w-full rounded-xl border border-slate-700/80 bg-navy-950 px-3.5 py-2.5 text-sm text-white transition focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              >
                {DEMO_ACCOUNTS.map((acc) => (
                  <option key={acc.id} value={acc.id}>
                    {acc.label} (${acc.balance.toLocaleString()})
                  </option>
                ))}
              </select>
            </div>

            {/* 4. Receiver Account */}
            <div>
              <label
                htmlFor="tx-receiver"
                className="block text-xs font-medium uppercase tracking-wider text-slate-300"
              >
                Receiver Account
              </label>
              <select
                id="tx-receiver"
                value={receiverAccount}
                onChange={(e) => setReceiverAccount(e.target.value)}
                className="mt-1.5 w-full rounded-xl border border-slate-700/80 bg-navy-950 px-3.5 py-2.5 text-sm text-white transition focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              >
                {DEMO_ACCOUNTS.map((acc) => (
                  <option key={acc.id} value={acc.id}>
                    {acc.label} (${acc.balance.toLocaleString()})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Collapsible Advanced Transaction Details */}
          <div className="rounded-xl border border-slate-800/90 bg-navy-950/60">
            <button
              type="button"
              onClick={() => setShowAdvanced((prev) => !prev)}
              className="flex w-full items-center justify-between px-4 py-3 text-left text-xs font-medium text-slate-300 hover:text-white transition"
            >
              <span>Advanced Transaction Details (Optional Overrides)</span>
              {showAdvanced ? (
                <ChevronUp className="h-4 w-4 text-slate-400" />
              ) : (
                <ChevronDown className="h-4 w-4 text-slate-400" />
              )}
            </button>

            {showAdvanced && (
              <div className="border-t border-slate-800/80 p-4 space-y-3">
                <p className="text-[11px] text-slate-400">
                  Leave blank to use demo account balances and automatically derived timestamps.
                </p>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div>
                    <label className="block text-[11px] font-medium text-slate-400">
                      Sender Balance Override
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      value={customOriginBalance}
                      onChange={(e) => setCustomOriginBalance(e.target.value)}
                      placeholder={senderDemo ? String(senderDemo.balance) : '0.00'}
                      className="mt-1 w-full rounded-lg border border-slate-700/80 bg-navy-900 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-400">
                      Receiver Balance Override
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      value={customDestBalance}
                      onChange={(e) => setCustomDestBalance(e.target.value)}
                      placeholder={receiverDemo ? String(receiverDemo.balance) : '0.00'}
                      className="mt-1 w-full rounded-lg border border-slate-700/80 bg-navy-900 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-400">
                      Transaction ID Override
                    </label>
                    <input
                      type="text"
                      value={customTxId}
                      onChange={(e) => setCustomTxId(e.target.value)}
                      placeholder="Auto-generated"
                      className="mt-1 w-full rounded-lg border border-slate-700/80 bg-navy-900 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-400">
                      Step (1+)
                    </label>
                    <input
                      type="number"
                      min="1"
                      value={customStep}
                      onChange={(e) => setCustomStep(e.target.value)}
                      placeholder="Auto"
                      className="mt-1 w-full rounded-lg border border-slate-700/80 bg-navy-900 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-400">
                      Day (0–30)
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="30"
                      value={customDay}
                      onChange={(e) => setCustomDay(e.target.value)}
                      placeholder="Auto"
                      className="mt-1 w-full rounded-lg border border-slate-700/80 bg-navy-900 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-400">
                      Hour (0–23)
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="23"
                      value={customHour}
                      onChange={(e) => setCustomHour(e.target.value)}
                      placeholder="Auto"
                      className="mt-1 w-full rounded-lg border border-slate-700/80 bg-navy-900 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {error && (
            <div className="flex items-start gap-2.5 rounded-xl border border-rose-500/40 bg-rose-500/10 p-3.5 text-xs text-rose-300">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-500 px-4 py-3 text-sm font-semibold text-navy-950 shadow-lg shadow-cyan-500/20 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Evaluating Transaction...
              </>
            ) : (
              <>
                <Send className="h-4 w-4" />
                Submit Transaction
              </>
            )}
          </button>
        </form>

        {/* Simplified Fraud Results Dashboard */}
        <div className="rounded-2xl border border-slate-800/90 bg-navy-900/95 p-6 shadow-xl lg:col-span-6">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
            <h3 className="text-sm font-semibold text-white">Risk Assessment Result</h3>
            {result && (
              <span className="font-mono text-xs text-slate-400">
                {result.transaction_id}
              </span>
            )}
          </div>

          {!result ? (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-slate-800 bg-navy-950 text-cyan-400/70">
                <ShieldCheck className="h-7 w-7" />
              </div>
              <p className="mt-4 text-sm font-medium text-slate-200">
                Ready for Evaluation
              </p>
              <p className="mt-1 max-w-xs text-xs text-slate-400">
                Enter the payment details on the left and submit to view the real-time fraud decision.
              </p>
            </div>
          ) : (
            <div className="mt-5 space-y-5 animate-fadeIn">
              {/* Prominent Decision & Risk Level */}
              <div
                className={`rounded-xl border p-4 ${
                  TONE_CLASSES[decisionMeta.decisionTone].card
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div
                      className={`flex h-11 w-11 items-center justify-center rounded-xl border ${
                        TONE_CLASSES[decisionMeta.decisionTone].badge
                      }`}
                    >
                      <decisionMeta.DecisionIcon className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                        Decision
                      </p>
                      <p
                        className={`text-xl font-bold ${
                          TONE_CLASSES[decisionMeta.decisionTone].text
                        }`}
                      >
                        {decisionMeta.decisionLabel}
                      </p>
                    </div>
                  </div>

                  <div className="text-right">
                    <p className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                      Risk Level
                    </p>
                    <span
                      className={`mt-1 inline-flex items-center rounded-full border px-3 py-0.5 text-xs font-semibold ${
                        TONE_CLASSES[decisionMeta.riskTone].badge
                      }`}
                    >
                      {decisionMeta.riskLevel} Risk
                    </span>
                  </div>
                </div>
              </div>

              {/* Fraud Probability Bar */}
              <div className="rounded-xl border border-slate-800/80 bg-navy-950/70 p-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-300">
                    Fraud Probability
                  </span>
                  <span className="text-lg font-bold text-white">
                    {(result.fraud_probability * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="mt-2.5 h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      TONE_CLASSES[decisionMeta.riskTone].bar
                    }`}
                    style={{
                      width: `${Math.min(100, Math.max(2, result.fraud_probability * 100))}%`,
                    }}
                  />
                </div>
              </div>

              {/* Explanation */}
              <div className="rounded-xl border border-slate-800/80 bg-navy-950/70 p-4">
                <p className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
                  Decision Explanation
                </p>
                <p className="mt-1.5 text-sm leading-relaxed text-slate-200">
                  {result.recommendation}
                </p>
              </div>

              {/* Additional Authentication Indicator */}
              <div className="flex items-center justify-between rounded-xl border border-slate-800/80 bg-navy-950/70 px-4 py-3">
                <div className="flex items-center gap-2.5">
                  <KeyRound className="h-4 w-4 text-cyan-400" />
                  <span className="text-xs font-medium text-slate-300">
                    Additional Authentication
                  </span>
                </div>
                {result.step_up_mfa_required ? (
                  <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-0.5 text-xs font-semibold text-amber-300">
                    <UserCheck className="h-3.5 w-3.5" />
                    MFA Required
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-800/70 px-2.5 py-0.5 text-xs font-medium text-slate-300">
                    Not Required
                  </span>
                )}
              </div>

              {/* Collapsible Technical Details */}
              <div className="rounded-xl border border-slate-800/90 bg-navy-950/50">
                <button
                  type="button"
                  onClick={() => setShowTechnicalDetails((prev) => !prev)}
                  className="flex w-full items-center justify-between px-4 py-3 text-left text-xs font-medium text-slate-300 hover:text-white transition"
                >
                  <span>Technical Details</span>
                  {showTechnicalDetails ? (
                    <ChevronUp className="h-4 w-4 text-slate-400" />
                  ) : (
                    <ChevronDown className="h-4 w-4 text-slate-400" />
                  )}
                </button>

                {showTechnicalDetails && (
                  <div className="border-t border-slate-800/80 p-4 grid grid-cols-1 gap-3 sm:grid-cols-2 text-xs">
                    <div className="rounded-lg border border-slate-800/70 bg-navy-900/70 p-3">
                      <p className="text-slate-400">ML Score / Threshold</p>
                      <p className="mt-1 font-mono font-semibold text-white">
                        {result.fraud_probability.toFixed(6)} /{' '}
                        {Number(result.model_threshold).toFixed(4)}
                      </p>
                    </div>
                    <div className="rounded-lg border border-slate-800/70 bg-navy-900/70 p-3">
                      <p className="flex items-center gap-1.5 text-slate-400">
                        <Cpu className="h-3.5 w-3.5 text-cyan-400" />
                        Model Name
                      </p>
                      <p className="mt-1 font-mono font-semibold text-white">
                        {result.model_version}
                      </p>
                    </div>
                    <div className="rounded-lg border border-slate-800/70 bg-navy-900/70 p-3">
                      <p className="flex items-center gap-1.5 text-slate-400">
                        <Clock className="h-3.5 w-3.5 text-cyan-400" />
                        Prediction Latency
                      </p>
                      <p className="mt-1 font-mono font-semibold text-white">
                        {Number(result.ml_latency_ms).toFixed(2)} ms
                      </p>
                    </div>
                    <div className="rounded-lg border border-slate-800/70 bg-navy-900/70 p-3">
                      <p className="flex items-center gap-1.5 text-slate-400">
                        <Database className="h-3.5 w-3.5 text-cyan-400" />
                        Database Transaction ID
                      </p>
                      <p className="mt-1 truncate font-mono font-semibold text-white" title={result.id}>
                        {result.id}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
