import React, { useState } from 'react'

import { Card } from 'antd'

import PagePanel from '@/components/PagePanel'
import { observer } from '@/hooks'

import ProjectTable from './components/ProjectTable'

const Project = () => {
  const data = [
    {
      id: 1,
      name: 'Project_00001'
    },
    {
      id: 2,
      name: 'Project_00002'
    },
    {
      id: 3,
      name: 'Project_00003'
    },
    {
      id: 4,
      name: 'Project_00004'
    }
  ]
  return (
    <PagePanel title="Project List">
      <Card style={{ minHeight: '100%' }}>
        <ProjectTable data={data} />
      </Card>
    </PagePanel>
  )
}

export default observer(Project)
