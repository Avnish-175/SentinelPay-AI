import React from 'react'
import { Link } from 'react-router-dom'
import { Database, History, Info, PlusCircle, Trash2 } from 'lucide-react'
import { useTransactions } from '../context/TransactionContext'
import ActionBadge from '../components/ActionBadge'

export default function TransactionHistoryPage() {
  const { sessionTransactions, clearSessionTransactions } = useTransactions()

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-xl font-bold text-white">Transaction History</h2>
          <p className="text-sm text-slate-400">
            Real persisted transaction responses received from Spring Boot during this session.
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          {sessionTransactions.length > 0 && (
            <button
              type="button"
              onClick={clearSessionTransactions}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-navy-800 px-3.5 py-2 text-xs font-medium text-slate-300 transition hover:bg-navy-700 hover:text-white"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Clear Session View
            </button>
          )}
          <Link
            to="/transactions/new"
            className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-navy-950 transition hover:bg-cyan-400"
          >
            <PlusCircle className="h-4 w-4" />
            New Transaction
          </Link>
        </div>
      </div>

      {/* Backend API Notice */}
      <div className="flex items-start gap-3 rounded-xl border border-cyan-500/30 bg-cyan-500/10 p-4 text-xs text-cyan-200">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-cyan-400" />
        <div>
          <p className="font-semibold text-cyan-100">
            Live Persistence Notice
          </p>
          <p className="mt-0.5 text-cyan-200/80">
            The Spring Boot backend currently exposes <code className="font-mono">POST /api/v1/transactions</code> for
            transaction scoring and PostgreSQL persistence, with no <code className="font-mono">GET /api/v1/transactions</code> history
            query endpoint yet. Only real transactions persisted during this active session are displayed below.
          </p>
        </div>
      </div>

      {sessionTransactions.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-slate-700/80 bg-navy-900/60 p-12 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full border border-slate-700 bg-navy-800 text-slate-400">
            <History className="h-6 w-6" />
          </div>
          <h3 className="mt-4 text-base font-semibold text-white">
            No persisted transactions in this session
          </h3>
          <p className="mx-auto mt-1.5 max-w-md text-xs text-slate-400">
            Submit a transaction on the New Transaction page to persist it to PostgreSQL
            and inspect its record here.
          </p>
        </div>
      ) : (
        <div className="rounded-xl border border-slate-800/90 bg-navy-900 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-slate-800 bg-navy-950/70 text-xs uppercase text-slate-400">
                <tr>
                  <th className="px-4 py-3.5">DB UUID / Tx ID</th>
                  <th className="px-4 py-3.5">Type</th>
                  <th className="px-4 py-3.5">Amount</th>
                  <th className="px-4 py-3.5">Fraud Prob</th>
                  <th className="px-4 py-3.5">Action</th>
                  <th className="px-4 py-3.5">MFA / Review</th>
                  <th className="px-4 py-3.5">Created At</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {sessionTransactions.map((tx) => (
                  <tr key={tx.id} className="hover:bg-navy-800/40">
                    <td className="px-4 py-3.5">
                      <div className="flex items-center gap-1.5 font-mono text-xs text-slate-200">
                        <Database className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                        <span>{tx.transaction_id || '—'}</span>
                      </div>
                      <div className="mt-0.5 font-mono text-[11px] text-slate-500">
                        {tx.id}
                      </div>
                    </td>
                    <td className="px-4 py-3.5 font-medium text-slate-200">{tx.type}</td>
                    <td className="px-4 py-3.5 font-medium text-white">
                      ${Number(tx.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td className="px-4 py-3.5 font-mono text-xs text-slate-200">
                      {(tx.fraud_probability * 100).toFixed(4)}%
                    </td>
                    <td className="px-4 py-3.5">
                      <ActionBadge action={tx.action} />
                    </td>
                    <td className="px-4 py-3.5 text-xs text-slate-300">
                      <div>MFA: {tx.step_up_mfa_required ? 'Yes' : 'No'}</div>
                      <div className="text-slate-400">
                        Review: {tx.manual_review_required ? 'Yes' : 'No'}
                      </div>
                    </td>
                    <td className="px-4 py-3.5 font-mono text-xs text-slate-400">
                      {tx.created_at ? new Date(tx.created_at).toLocaleString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
