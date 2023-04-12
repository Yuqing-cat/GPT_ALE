import React, { useRef, useState } from 'react'

import { Button, Card, message, Space } from 'antd'
import { useQuery } from 'react-query'
import { useParams } from 'react-router'

import API from '@/api'
import PagePanel from '@/components/PagePanel'
import { observer } from '@/hooks'

import LineChart, { LineChartHandle } from './components/LineChart'

const ProjectDetail = () => {
  const { name } = useParams()

  const chartRef = useRef<LineChartHandle>(null)

  const { isLoading, data } = useQuery<any[]>(
    ['project detail'],
    async () => {
      try {
        return await API.palantir.getLineCharts(name || '')
      } catch (e: any) {
        message.error(e.message)
        return null
      }
    },
    {
      retry: false,
      refetchOnWindowFocus: false
    }
  )
  return (
    <PagePanel title={`Project: ${name}`}>
      <Card
        extra={
          <Space>
            <Button
              type="primary"
              onClick={() => {
                chartRef.current?.legendAllSelect()
              }}
            >
              Select All
            </Button>
            <Button
              ghost
              type="primary"
              onClick={() => {
                chartRef.current?.legendInverseSelect()
              }}
            >
              Inverse Select
            </Button>
          </Space>
        }
        title="Model Performance"
      >
        <LineChart ref={chartRef} loading={isLoading} data={data} />
      </Card>
    </PagePanel>
  )
}

export default observer(ProjectDetail)
