import React, { useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  Activity,
  History,
  LayoutDashboard,
  Menu,
  PlusCircle,
  ShieldCheck,
  X,
} from 'lucide-react'
import { API_BASE_URL } from '../services/api'

const navItems = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/transactions/new', label: 'New Transaction', icon: PlusCircle },
  { to: '/transactions/history', label: 'Transaction History', icon: History },
]

export default function Layout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="min-h-screen bg-navy-950 text-slate-100 flex">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-64 transform border-r border-slate-800/80 bg-navy-900 transition-transform duration-200 ease-in-out lg:static lg:translate-x-0 flex flex-col ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex h-16 items-center justify-between border-b border-slate-800/80 px-5">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-cyan-500/15 border border-cyan-400/30 text-cyan-400">
              <ShieldCheck className="h-5 w-5" />
            </div>
            <div>
              <span className="text-base font-bold tracking-tight text-white">
                SentinelPay
              </span>
              <span className="ml-1 rounded bg-cyan-500/20 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-cyan-300">
                AI
              </span>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setSidebarOpen(false)}
            className="rounded-md p-1 text-slate-400 hover:bg-navy-800 hover:text-white lg:hidden"
            aria-label="Close navigation"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <nav className="flex-1 space-y-1.5 px-3 py-5">
          {navItems.map((item) => {
            const Icon = item.icon
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                onClick={() => setSidebarOpen(false)}
                className={({ isActive }) =>
                  `flex items-center gap-3 rounded-lg px-3.5 py-2.5 text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                      : 'text-slate-400 hover:bg-navy-800 hover:text-slate-100 border border-transparent'
                  }`
                }
              >
                <Icon className="h-4 w-4 shrink-0" />
                {item.label}
              </NavLink>
            )
          })}
        </nav>

        <div className="border-t border-slate-800/80 p-4">
          <div className="rounded-lg border border-slate-800 bg-navy-950/70 p-3">
            <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
              <Activity className="h-3.5 w-3.5 text-cyan-400" />
              Backend Gateway
            </div>
            <p className="mt-1 truncate font-mono text-[11px] text-slate-400" title={API_BASE_URL}>
              {API_BASE_URL}
            </p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex flex-1 flex-col min-w-0">
        <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-slate-800/80 bg-navy-900/80 px-4 backdrop-blur-md sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              className="rounded-lg border border-slate-800 bg-navy-800 p-2 text-slate-300 hover:bg-navy-700 hover:text-white lg:hidden"
              aria-label="Open navigation"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div>
              <h1 className="text-sm font-semibold text-white sm:text-base">
                Real-Time Fraud Intelligence Console
              </h1>
              <p className="hidden text-xs text-slate-400 sm:block">
                XGBoost Risk Scoring • Spring Boot Orchestration • PostgreSQL Ledger
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-navy-800 px-3 py-1 text-xs font-medium text-slate-300">
              <span className="h-2 w-2 rounded-full bg-cyan-400" />
              Live Scoring
            </span>
          </div>
        </header>

        <main className="flex-1 p-4 sm:p-6 lg:p-8">
          <div className="mx-auto max-w-7xl">{children}</div>
        </main>
      </div>
    </div>
  )
}
