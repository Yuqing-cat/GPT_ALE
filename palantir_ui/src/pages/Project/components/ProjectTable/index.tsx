import React from 'react'

import { Tag } from 'antd'
import dayjs from 'dayjs'
import { number } from 'echarts'
import { Link } from 'react-router-dom'

import ResizeTable, { ResizeColumnType } from '@/components/ResizeTable'

export interface JobTableProps {
  loading?: boolean
  data?: any[]
}

export interface SearchModel {
  scope?: string
  roleName?: string
}

const JobTable = (props: JobTableProps) => {
  const { loading, data } = props

  const getDetialUrl = (name: number) => {
    return `/project/${name}`
  }

  const columns: ResizeColumnType<any>[] = [
    {
      title: 'name',
      dataIndex: 'name',
      ellipsis: true,
      width: 200,
      render: (col: number, record: any) => {
        return <Link to={getDetialUrl(col)}>{col}</Link>
      }
    },
    // {
    //   title: 'Status',
    //   dataIndex: 'status',
    //   width: 200
    // },
    {
      title: ''
    }
  ]

  return (
    <ResizeTable
      columns={columns}
      dataSource={data}
      loading={loading}
      rowKey="id"
      scroll={{ x: '100%' }}
    />
  )
}

export default JobTable
