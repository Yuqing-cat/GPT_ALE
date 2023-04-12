import React, { useEffect } from 'react'

import {
  BellOutlined,
  CompressOutlined,
  ExpandOutlined,
  LogoutOutlined,
  UserOutlined
} from '@ant-design/icons'
import { useAccount, useMsal } from '@azure/msal-react'
import { Layout, Space, Dropdown, Avatar, Badge } from 'antd'
import { Link } from 'react-router-dom'

import { useFullScreen, observer, useStore } from '@/hooks'

import JobsProgress from '../JobsProgress'

import styles from './index.module.less'
const { Header } = Layout

const HeaderBar = () => {
  const { globalStore } = useStore()
  const { jobsCount, getJobsProgress } = globalStore

  const { instance, accounts } = useMsal()
  const account = useAccount(accounts[0] || {})
  const { fullScreen, toggleFullScreen } = useFullScreen()

  const onClickMenu = ({ key }: any) => {
    switch (key) {
      case 'logout':
        instance.logoutRedirect().catch((e) => {
          console.error(e)
        })
        break
      default:
        break
    }
  }

  useEffect(() => {
    getJobsProgress()
    setInterval(() => {
      getJobsProgress()
    }, 30000)
  }, [])

  return (
    <>
      <Header className={styles.header}>
        <div className={styles.logoBar}>
          <Link to="/">
            <img alt="logo" src="/logo200.png" />
            <h1>Palantir</h1>
          </Link>
        </div>
        <Space className={styles.right} size={0}>
          {/* <span className={styles.searchBar}>
            <Input
              allowClear
              placeholder="search..."
              prefix={<SearchOutlined />}
              onPressEnter={(e) => {
                const { value } = e.target as HTMLInputElement
                onSearch?.(value)
              }}
            />
          </span> */}

          <span className={styles.action} onClick={toggleFullScreen}>
            {fullScreen ? (
              <CompressOutlined style={{ fontSize: 16 }} />
            ) : (
              <ExpandOutlined style={{ fontSize: 16 }} />
            )}
          </span>

          <Dropdown
            dropdownRender={() => {
              return <JobsProgress />
            }}
            trigger={['click']}
          >
            <span className={styles.action}>
              <Badge count={jobsCount} size="small">
                <BellOutlined style={{ color: '#fff', fontSize: 16 }} />
              </Badge>
            </span>
          </Dropdown>
          {account?.username && (
            <Dropdown
              menu={{
                className: styles.menu,
                items: [
                  {
                    key: 'logout',
                    icon: <LogoutOutlined />,
                    label: 'Logout'
                  }
                ],
                onClick: onClickMenu
              }}
              placement="bottomLeft"
            >
              <span className={`${styles.action} ${styles.account}`}>
                <Avatar className={styles.avatar} icon={<UserOutlined />} size="small" />
                <span>{account.username}</span>
              </span>
            </Dropdown>
          )}
        </Space>
      </Header>
      <div className={styles.vacancy} />
    </>
  )
}

export default observer(HeaderBar)
