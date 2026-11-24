import React from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { TransactionProvider } from './context/TransactionContext'
import Layout from './components/Layout'
import OverviewPage from './pages/OverviewPage'
import NewTransactionPage from './pages/NewTransactionPage'
import TransactionHistoryPage from './pages/TransactionHistoryPage'

export default function App() {
  return (
    <TransactionProvider>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/transactions/new" element={<NewTransactionPage />} />
            <Route path="/transactions/history" element={<TransactionHistoryPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </TransactionProvider>
  )
}
