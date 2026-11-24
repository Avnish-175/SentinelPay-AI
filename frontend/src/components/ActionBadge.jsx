import React from 'react'
import { CheckCircle2, ShieldAlert, XOctagon } from 'lucide-react'

export default function ActionBadge({ action }) {
  if (action === 'APPROVE') {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 text-xs font-semibold text-emerald-400">
        <CheckCircle2 className="h-3.5 w-3.5" />
        APPROVE
      </span>
    )
  }
  if (action === 'CHALLENGE') {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-1 text-xs font-semibold text-amber-400">
        <ShieldAlert className="h-3.5 w-3.5" />
        CHALLENGE
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-rose-500/30 bg-rose-500/10 px-2.5 py-1 text-xs font-semibold text-rose-400">
      <XOctagon className="h-3.5 w-3.5" />
      BLOCK
    </span>
  )
}
