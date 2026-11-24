import axios from 'axios'

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8081'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

/**
 * Submit a transaction to the Spring Boot backend for ML scoring and risk evaluation.
 * Endpoint: POST /api/v1/transactions
 */
export async function submitTransaction(payload) {
  const response = await apiClient.post('/api/v1/transactions', payload)
  return response.data
}

export default apiClient
