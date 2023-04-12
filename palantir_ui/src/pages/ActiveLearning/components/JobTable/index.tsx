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

  const getDetialUrl = (id: number) => {
    return `/activeLearning/job/${id}`
  }

  const columns: ResizeColumnType<any>[] = [
    {
      key: 'job',
      title: 'Job',
      dataIndex: 'jobid',
      ellipsis: true,
      width: 200,
      render: (col: number, record: any) => {
        return <Link to={getDetialUrl(col)}>{col}</Link>
      }
    },
    {
      key: 'Date',
      title: 'Date',
      dataIndex: 'creation_time',
      width: 200,
      render: (col: string, record: any) => {
        return dayjs(col).format('YYYY-MM-DD HH:mm:ss')
      }
    },
    {
      key: 'Number Categories',
      title: 'Number Categories',
      dataIndex: 'number_of_category',
      width: 200
    },
    {
      key: 'Can Annotate',
      title: 'Can Annotate',
      dataIndex: 'can_annotate',
      width: 200,
      render: (col: boolean) => {
        return <Tag color={col ? 'processing' : 'default'}>{col ? 'True' : 'False'}</Tag>
      }
    },
    {
      key: 'Model Accuracy',
      title: 'Model Accuracy',
      dataIndex: 'acc',
      width: 200,
      render: (col: string) => {
        return isNaN(Number(col)) ? col : `${col}%`
      }
    },
    {
      key: 'Model Precision',
      title: 'Model Precision',
      dataIndex: 'precision',
      width: 200,
      render: (col: string) => {
        return isNaN(Number(col)) ? col : `${col}%`
      }
    },
    {
      key: 'Model Recall',
      title: 'Model Recall',
      dataIndex: 'recall',
      width: 200,
      render: (col: string) => {
        return isNaN(Number(col)) ? col : `${col}%`
      }
    },
    {
      key: 'Model F1',
      title: 'Model F1',
      dataIndex: 'f1',
      width: 200,
      render: (col: string) => {
        return isNaN(Number(col)) ? col : `${col}%`
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
      rowKey="jobid"
      scroll={{ x: '100%' }}
    />
  )
}

export default JobTable
