import React, { useState } from 'react'

import { Card, message } from 'antd'
import dayjs from 'dayjs'
import { useQuery } from 'react-query'

import API from '@/api'
import { AllRunsRespose } from '@/api/gpt_ale/interface'
import PagePanel from '@/components/PagePanel'
import { observer } from '@/hooks'

import JobTable from './components/JobTable'
import SearchBar from './components/SearchBar'

const ActiveLearning = () => {
  // const { globalStore } = useStore()
  const { isLoading, data, refetch } = useQuery<any[]>(
    ['activeLearning'],
    async () => {
      try {
        const result = await API.gpt_ale.allRuns()
        return result.sort((a: AllRunsRespose, b: AllRunsRespose) => {
          return dayjs(b.creation_time).diff(dayjs(a.creation_time), 'milliseconds', true)
        })
      } catch (e: any) {
        message.error(e.message)
        return []
      }
    },
    {
      retry: false,
      refetchOnWindowFocus: false
      // refetchInterval: 5000
    }
  )

  return (
    <PagePanel title="Annotation Jobs">
      <Card style={{ minHeight: '100%' }}>
        <SearchBar onSearch={() => refetch} />
        <JobTable loading={isLoading} data={data} />
      </Card>
    </PagePanel>
  )
}

export default observer(ActiveLearning)
