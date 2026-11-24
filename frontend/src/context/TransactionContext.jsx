import React, { createContext, useContext, useMemo, useState } from 'react'

const TransactionContext = createContext(null)

export function TransactionProvider({ children }) {
  // Stores real transaction responses returned by Spring Boot POST /api/v1/transactions
  // during this active browser session (no fabricated or mocked records).
  const [sessionTransactions, setSessionTransactions] = useState([])
  const [lastError, setLastError] = useState(null)

  const addTransactionRecord = (record) => {
    setSessionTransactions((prev) => [record, ...prev])
    setLastError(null)
  }

  const clearSessionTransactions = () => {
    setSessionTransactions([])
  }

  const stats = useMemo(() => {
    const total = sessionTransactions.length
    if (total === 0) {
      return {
        total: 0,
        approved: 0,
        challenged: 0,
        blocked: 0,
        avgProbability: null,
        avgLatencyMs: null,
        totalVolume: 0,
        actionDistribution: [],
        typeDistribution: [],
      }
    }

    let approved = 0
    let challenged = 0
    let blocked = 0
    let probSum = 0
    let latencySum = 0
    let latencyCount = 0
    let totalVolume = 0
    const typeCounts = {}

    for (const tx of sessionTransactions) {
      if (tx.action === 'APPROVE') approved += 1
      else if (tx.action === 'CHALLENGE') challenged += 1
      else if (tx.action === 'BLOCK') blocked += 1

      if (typeof tx.fraud_probability === 'number') {
        probSum += tx.fraud_probability
      }
      if (typeof tx.ml_latency_ms === 'number') {
        latencySum += tx.ml_latency_ms
        latencyCount += 1
      }
      if (typeof tx.amount === 'number') {
        totalVolume += tx.amount
      }
      if (tx.type) {
        typeCounts[tx.type] = (typeCounts[tx.type] || 0) + 1
      }
    }

    const actionDistribution = [
      { name: 'APPROVE', count: approved, fill: '#10b981' },
      { name: 'CHALLENGE', count: challenged, fill: '#f59e0b' },
      { name: 'BLOCK', count: blocked, fill: '#ef4444' },
    ]

    const typeDistribution = Object.entries(typeCounts).map(([type, count]) => ({
      type,
      count,
    }))

    return {
      total,
      approved,
      challenged,
      blocked,
      avgProbability: probSum / total,
      avgLatencyMs: latencyCount > 0 ? latencySum / latencyCount : null,
      totalVolume,
      actionDistribution,
      typeDistribution,
    }
  }, [sessionTransactions])

  const value = {
    sessionTransactions,
    addTransactionRecord,
    clearSessionTransactions,
    stats,
    lastError,
    setLastError,
  }

  return (
    <TransactionContext.Provider value={value}>
      {children}
    </TransactionContext.Provider>
  )
}

export function useTransactions() {
  const ctx = useContext(TransactionContext)
  if (!ctx) {
    throw new Error('useTransactions must be used within a TransactionProvider')
  }
  return ctx
}
