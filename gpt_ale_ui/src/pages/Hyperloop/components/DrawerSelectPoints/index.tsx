import React, {
  forwardRef,
  MutableRefObject,
  useContext,
  useEffect,
  useImperativeHandle,
  useRef,
  useState
} from 'react'

import { DownOutlined } from '@ant-design/icons'
import { Badge, Button, Drawer, Dropdown, Space, Table, InputNumber, message } from 'antd'
import type { TableProps } from 'antd'
import type { SorterResult } from 'antd/es/table/interface'
import { uniqBy } from 'lodash-es'

import { ChoicePoint } from '@/api/gpt_ale/interface'
import useForceUpdate from '@/hooks/useForceUpdate'
import { getColor } from '@/utils/color'

import { HyperloopContext } from '../HyperloopContext'
import ToolTip, { TooltipInstance } from '../Tooltip'

export interface DrawerSelectPointsHandle {
  open: () => void
  close: () => void
  forceUpdate: () => void
}

export interface DrawerSelectPointsProps {
  score?: number
  points: MutableRefObject<ChoicePoint[]>
  onConfirm: (points: ChoicePoint[]) => void
}

const items = [{ key: '1', label: 'Set all points gpt' }]

const DrawerSelectPoints = (props: DrawerSelectPointsProps, ref: any) => {
  const { categories, onCreateCategory } = useContext(HyperloopContext)

  const { points, onConfirm } = props
  const fillerPoints = useRef<ChoicePoint[]>([])
  const tooltipRef = useRef<TooltipInstance>(null)
  const [toolTipData, setToolTipData] = useState<any>({})
  const divRef = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState<boolean>(false)
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([])
  const [sortedInfo, setSortedInfo] = useState<SorterResult<any>>({})
  const forceUpdate = useForceUpdate()
  const [confidenceThresholdValue, setConfidenceThresholdValue] = useState<number | null>(0.00001)

  const handleChange: TableProps<any>['onChange'] = (pagination, filters, sorter) => {
    setSortedInfo(sorter as SorterResult<any>)
  }

  const columns: any[] = [
    {
      title: 'Score',
      dataIndex: 'score',
      width: 90,
      sorter: {
        compare: (a: any, b: any) => a.score - b.score,
        multiple: 1
      },
      sortOrder: sortedInfo.field === 'score' ? sortedInfo.order : null,
      render: (col: number) => {
        return col.toFixed(5)
      }
    },
    {
      title: 'Sentence',
      dataIndex: 'text'
    },
    {
      title: 'Ann By',
      dataIndex: 'ann_by',
      width: 100
    },
    {
      title: 'Category',
      dataIndex: 'newLabel',
      width: 220,
      render: (col: string, record: any) => {
        return (
          <Button
            onClick={(e) => {
              let elm = e.target as HTMLElement | null | undefined
              while (elm && elm?.nodeName !== 'BUTTON') {
                elm = elm?.parentElement
              }

              const rect = elm?.getBoundingClientRect()

              let top = rect!.bottom + 10

              if (window.innerHeight < top + 370) {
                top = rect!.top - 380
              }

              divRef.current?.setAttribute(
                'style',
                `position: fixed; 
                 z-index: 9999999;
                 top: ${top}px;
                 right: 380px;
                `
              )
              tooltipRef.current?.show()
              setToolTipData(record)
            }}
          >
            <Badge color={getColor(col)} text={col} />
          </Button>
        )
      }
    }
  ]

  const onClose = () => {
    setOpen(false)
  }

  const onUpdate = (point: ChoicePoint, name: string) => {
    point.newLabel = name

    setSelectedRowKeys((state) => {
      if (!state.includes(point.id)) {
        state.push(point.id)
      }
      return [...state]
    })
  }

  const onCreate = (name: string) => {
    onCreateCategory?.([name])
    onUpdate(toolTipData, name)
    tooltipRef.current?.hide()
  }

  const onClickConfirm = () => {
    const list: ChoicePoint[] = selectedRowKeys.map((id) => {
      const item = { ...(fillerPoints.current.find((item) => item.id === id)! as ChoicePoint) }
      if (!item.newLabel) {
        item.newLabel = item.label
      }
      return item
    })
    onConfirm(list)
    setSelectedRowKeys([])
  }

  const onSelectChange = (newSelectedRowKeys: React.Key[]) => {
    setSelectedRowKeys(newSelectedRowKeys)
  }

  const onChangeConfidenceThresholdInput = (value: number | null = 0) => {
    setConfidenceThresholdValue(value)
  }

  const onMenuClick = () => {
    const keys: React.Key[] = []
    const labels = new Set<string>([])
    fillerPoints.current.forEach((item) => {
      if (item.gpt !== 'unknown' && item.gpt !== '') {
        item.newLabel = item.gpt
        labels.add(item.gpt!)
        keys.push(item.id)
      }
    })
    message.open({
      content: (
        <div style={{ width: 350 }}>
          selected [{fillerPoints.current.length}] points to accept GPT suggestions, [
          {fillerPoints.current.length - keys.length}] of them are unknown or blank,
          <br /> skip these points, only [{keys.length}] points are valid and confirmed.
        </div>
      )
    })
    setSelectedRowKeys((state) => {
      return Array.from(keys)
    })
    onCreateCategory?.(Array.from<string>(labels))
  }

  useImperativeHandle<any, DrawerSelectPointsHandle>(
    ref,
    () => {
      return {
        open: () => {
          setSelectedRowKeys(fillerPoints.current.map((item) => item.id))
          setSortedInfo({})
          setOpen(true)
        },
        close: onClose,
        forceUpdate
      }
    },
    []
  )

  useEffect(() => {
    fillerPoints.current = uniqBy(points.current, 'id').filter((item) => {
      return item.score >= (confidenceThresholdValue || 0)
    })
    forceUpdate()
  }, [confidenceThresholdValue, points.current])

  return (
    <>
      <Drawer
        title={
          <div
            style={{
              width: '100%',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <Space>
              Confidence Threshold:
              <InputNumber
                value={confidenceThresholdValue}
                step="0.00001"
                max={1}
                onChange={onChangeConfidenceThresholdInput}
              />
            </Space>
            <span>
              ({selectedRowKeys.length} of {fillerPoints.current.length} selected)
            </span>
            <Space>
              <Button type="primary" onClick={onClickConfirm}>
                Confirm
              </Button>
              <Dropdown menu={{ items, onClick: onMenuClick }}>
                <Button type="default">
                  More <DownOutlined />
                </Button>
              </Dropdown>
            </Space>
          </div>
        }
        size="large"
        open={open}
        placement="right"
        bodyStyle={{ padding: 0 }}
        onClose={onClose}
      >
        <Table
          rowSelection={{
            columnWidth: 75,
            fixed: true,
            selectedRowKeys,
            onChange: onSelectChange
          }}
          rowKey="id"
          columns={columns}
          dataSource={fillerPoints.current}
          pagination={false}
          scroll={{ y: 'calc(100vh - 130px)' }}
          onChange={handleChange}
        />
      </Drawer>

      <div ref={divRef} style={{ display: 'none' }}>
        <div id="ddd"></div>
      </div>

      <ToolTip
        ref={tooltipRef}
        elmId="ddd"
        showText={false}
        categories={categories}
        data={toolTipData}
        onChange={onUpdate}
        onCreate={onCreate}
      />
    </>
  )
}

export default forwardRef<DrawerSelectPointsHandle, DrawerSelectPointsProps>(DrawerSelectPoints)
