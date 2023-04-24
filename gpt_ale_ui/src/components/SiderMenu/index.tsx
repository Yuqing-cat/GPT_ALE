import React, { useEffect, useRef, useState } from 'react'

import {
  HomeOutlined,
  FormOutlined,
  ProfileOutlined,
  TableOutlined,
  WarningOutlined,
  BuildOutlined,
  RocketOutlined
} from '@ant-design/icons'
import { Layout, Menu } from 'antd'
import { Link, useLocation } from 'react-router-dom'

// import VersionBar from '../FooterBar/VersionBar'

import styles from './index.module.less'

interface SiderMenuProps {
  collapsedWidth?: number
  siderWidth?: number
}

const { Sider } = Layout

const menuItems = [
  // {
  //   key: '/',
  //   icon: <HomeOutlined />,
  //   label: <Link to="/">Home</Link>
  // },
  // {
  //   key: '/list',
  //   icon: <TableOutlined />,
  //   label: <Link to="/list">List</Link>
  // },
  // {
  //   key: '/form',
  //   icon: <FormOutlined />,
  //   label: <Link to="/form">Form</Link>
  // },
  // {
  //   key: '/detail',
  //   icon: <ProfileOutlined />,
  //   label: <Link to="/detail">Detail</Link>
  // },
  // {
  //   key: 'exception',
  //   icon: <WarningOutlined />,
  //   label: 'Exception',
  //   children: [
  //     { key: '/403', label: <Link to="/403">403</Link> },
  //     { key: '/404', label: <Link to="/404">404</Link> }
  //   ]
  // },
  {
    key: '/project',
    icon: <RocketOutlined />,
    label: <Link to="/project">Project</Link>
  },
  {
    key: '/activeLearning',
    icon: <BuildOutlined />,
    label: <Link to="/activeLearning">Active Learning</Link>
  },
  {
    key: '/mlflow',
    icon: <BuildOutlined />,
    label: <Link to="/mlflow">MLflow</Link>
  }
]

const defaultProps = {
  collapsedWidth: 48,
  siderWidth: 200
}

const getMenuKey = (pathname: string) => {
  return pathname
}

function SiderMenu(props: SiderMenuProps) {
  const { siderWidth, collapsedWidth } = { ...defaultProps, ...props }
  const [collapsed, setCollapsed] = useState<boolean>(true)
  const [transition, setTransition] = useState<string>('none')
  const location = useLocation()

  const [current, setcurrent] = useState<string>(getMenuKey(location.pathname))

  useEffect(() => {
    setcurrent(getMenuKey(location.pathname))
  }, [location.pathname])

  useEffect(() => {
    setTransition('all 0.2s ease 0s')
  }, [])

  return (
    <>
      <div
        style={{
          width: collapsed ? collapsedWidth : siderWidth,
          overflow: 'hidden',
          flex: `0 0 ${collapsed ? collapsedWidth : siderWidth}px`,
          maxWidth: collapsed ? collapsedWidth : siderWidth,
          minWidth: collapsed ? collapsedWidth : siderWidth,
          transition
        }}
      />
      <Sider
        collapsible
        breakpoint="lg"
        className={styles.sider}
        collapsed={collapsed}
        collapsedWidth={collapsedWidth}
        theme="light"
        width={siderWidth}
        onCollapse={setCollapsed}
      >
        {/* <div className="logo" /> */}
        <div style={{ flex: '1 1 0%', overflow: 'hidden auto' }}>
          <Menu
            className={styles.menu}
            items={menuItems}
            mode="inline"
            selectedKeys={[current]}
            style={{ height: '100%', borderRight: 0 }}
          />
        </div>
      </Sider>
    </>
  )
}

export default SiderMenu
