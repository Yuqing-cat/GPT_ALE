import React from 'react'

import { LoadingOutlined } from '@ant-design/icons'
import { List, Progress, Space } from 'antd'

import { observer, useStore } from '@/hooks'

import styles from './index.module.less'

export interface JobsProgressProps {}

const JobsProgress = (props: JobsProgressProps) => {
  const { globalStore } = useStore()
  const { jobs } = globalStore

  return (
    <div className={styles.wrap}>
      {jobs.length > 0 && (
        <List
          renderItem={(item) => (
            <List.Item key={item.jobid}>
              <List.Item.Meta
                title={
                  <Space>
                    {item.jobid}
                    {item.jobid.includes('Running') && (
                      <LoadingOutlined style={{ color: '#1890ff' }} />
                    )}
                  </Space>
                }
                description={<Progress percent={item.progress || 0} />}
              />
            </List.Item>
          )}
          itemLayout="horizontal"
          dataSource={jobs}
        />
      )}
    </div>
  )
}

export default observer(JobsProgress)
