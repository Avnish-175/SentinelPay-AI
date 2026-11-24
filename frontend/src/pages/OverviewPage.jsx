import React from 'react'
import { Link } from 'react-router-dom'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Clock,
  DollarSign,
  PlusCircle,
  ShieldAlert,
  XOctagon,
} from 'lucide-react'
import { useTransactions } from '../context/TransactionContext'
import ActionBadge from '../components/ActionBadge'

function StatCard({ title, value, subtitle, icon: Icon, accent = 'cyan' }) {
  const accentStyles = {
    cyan: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20',
    emerald: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    amber: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    rose: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
  }

  return (
    <div className="rounded-xl border border-slate-800/90 bg-navy-900 p-5 shadow-lg shadow-black/20">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
            {title}
          </p>
          <p className="mt-2 text-2xl font-bold text-white">{value}</p>
          {subtitle && (
            <p className="mt-1 text-xs text-slate-400">{subtitle}</p>
          )}
        </div>
        <div className={`rounded-lg border p-2.5 ${accentStyles[accent]}`}>
          <Icon className="h-5 w-5" />
        </div>
      </div>
    </div>
  )
}

export default function OverviewPage() {
  const { stats, sessionTransactions } = useTransactions()

  if (stats.total === 0) {
    return (
      <div className="space-y-6">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-xl font-bold text-white">Risk Overview</h2>
            <p className="text-sm text-slate-400">
              Live statistics from transactions evaluated via Spring Boot &amp; FastAPI.
            </p>
          </div>
          <Link
            to="/transactions/new"
            className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-navy-950 transition hover:bg-cyan-400"
          >
            <PlusCircle className="h-4 w-4" />
            Evaluate Transaction
          </Link>
        </div>

        <div className="rounded-2xl border border-dashed border-slate-700/80 bg-navy-900/60 p-12 text-center">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full border border-cyan-500/30 bg-cyan-500/10 text-cyan-400">
            <Activity className="h-7 w-7" />
          </div>
          <h3 className="mt-4 text-lg font-semibold text-white">
            No transactions evaluated in this session yet
          </h3>
          <p className="mx-auto mt-2 max-w-md text-sm text-slate-400">
            Submit a real transaction through the New Transaction console to view live
            fraud probabilities, risk policy decisions, and session analytics.
          </p>
          <div className="mt-6">
            <Link
              to="/transactions/new"
              className="inline-flex items-center gap-2 rounded-lg border border-cyan-500/40 bg-cyan-500/15 px-5 py-2.5 text-sm font-semibold text-cyan-300 transition hover:bg-cyan-500/25"
            >
              Go to New Transaction
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-xl font-bold text-white">Risk Overview</h2>
          <p className="text-sm text-slate-400">
            Real-time metrics from {stats.total} persisted transaction{stats.total === 1 ? '' : 's'} in this session.
          </p>
        </div>
        <Link
          to="/transactions/new"
          className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-navy-950 transition hover:bg-cyan-400"
        >
          <PlusCircle className="h-4 w-4" />
          New Transaction
        </Link>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          title="Total Evaluated"
          value={stats.total}
          subtitle={`Volume: $${stats.totalVolume.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}
          icon={DollarSign}
          accent="cyan"
        />
        <StatCard
          title="Approved"
          value={stats.approved}
          subtitle={`${((stats.approved / stats.total) * 100).toFixed(1)}% of evaluated`}
          icon={CheckCircle2}
          accent="emerald"
        />
        <StatCard
          title="Challenged (MFA / Review)"
          value={stats.challenged}
          subtitle={`${((stats.challenged / stats.total) * 100).toFixed(1)}% of evaluated`}
          icon={ShieldAlert}
          accent="amber"
        />
        <StatCard
          title="Blocked"
          value={stats.blocked}
          subtitle={
            stats.avgLatencyMs !== null
              ? `Avg ML latency: ${stats.avgLatencyMs.toFixed(1)} ms`
              : 'Critical risk'
          }
          icon={XOctagon}
          accent="rose"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-800/90 bg-navy-900 p-5">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">
                Risk Policy Action Distribution
              </h3>
              <p className="text-xs text-slate-400">
                Breakdown of APPROVE, CHALLENGE, and BLOCK outcomes
              </p>
            </div>
            <span className="inline-flex items-center gap-1 rounded-md bg-navy-800 px-2.5 py-1 text-xs text-slate-300">
              <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
              Avg Prob: {(stats.avgProbability * 100).toFixed(2)}%
            </span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.actionDistribution} margin={{ top: 10, right: 16, left: -12, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                <YAxis allowDecimals={false} stroke="#94a3b8" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f1b33',
                    borderColor: '#334155',
                    borderRadius: '0.5rem',
                    color: '#f8fafc',
                  }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {stats.actionDistribution.map((entry) => (
                    <Cell key={entry.name} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border border-slate-800/90 bg-navy-900 p-5">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">
                Transaction Volume by Type
              </h3>
              <p className="text-xs text-slate-400">
                Distribution across payment channels
              </p>
            </div>
            <Clock className="h-4 w-4 text-slate-400" />
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={stats.typeDistribution}
                  dataKey="count"
                  nameKey="type"
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={88}
                  paddingAngle={4}
                  label={({ type, count }) => `${type} (${count})`}
                >
                  {stats.typeDistribution.map((entry, idx) => {
                    const colors = ['#06b6d4', '#3b82f6', '#8b5cf6', '#10b981', '#f59e0b']
                    return <Cell key={entry.type} fill={colors[idx % colors.length]} />
                  })}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f1b33',
                    borderColor: '#334155',
                    borderRadius: '0.5rem',
                    color: '#f8fafc',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Recent Evaluations Table */}
      <div className="rounded-xl border border-slate-800/90 bg-navy-900 overflow-hidden">
        <div className="flex items-center justify-between border-b border-slate-800/80 px-5 py-4">
          <h3 className="text-sm font-semibold text-white">Recent Evaluations</h3>
          <Link
            to="/transactions/history"
            className="text-xs font-medium text-cyan-400 hover:text-cyan-300"
          >
            View all →
          </Link>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-800 bg-navy-950/60 text-xs uppercase text-slate-400">
              <tr>
                <th className="px-5 py-3">Transaction ID</th>
                <th className="px-5 py-3">Type</th>
                <th className="px-5 py-3">Amount</th>
                <th className="px-5 py-3">Fraud Prob</th>
                <th className="px-5 py-3">Action</th>
                <th className="px-5 py-3">Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {sessionTransactions.slice(0, 5).map((tx) => (
                <tr key={tx.id} className="hover:bg-navy-800/40">
                  <td className="px-5 py-3 font-mono text-xs text-slate-300">
                    {tx.transaction_id || tx.id}
                  </td>
                  <td className="px-5 py-3 text-slate-200">{tx.type}</td>
                  <td className="px-5 py-3 font-medium text-white">
                    ${Number(tx.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </td>
                  <td className="px-5 py-3 font-mono text-xs text-slate-200">
                    {(tx.fraud_probability * 100).toFixed(4)}%
                  </td>
                  <td className="px-5 py-3">
                    <ActionBadge action={tx.action} />
                  </td>
                  <td className="px-5 py-3 font-mono text-xs text-slate-400">
                    {tx.ml_latency_ms != null ? `${tx.ml_latency_ms} ms` : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
