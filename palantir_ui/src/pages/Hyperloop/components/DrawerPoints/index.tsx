import React, {
  forwardRef,
  useContext,
  useEffect,
  useImperativeHandle,
  useRef,
  useState
} from 'react'

import { Badge, Button, Drawer, Space, Table } from 'antd'

import { ChoicePoint } from '@/api/palantir/interface'
import useForceUpdate from '@/hooks/useForceUpdate'
import { getColor } from '@/utils/color'

import { HyperloopContext } from '../HyperloopContext'
import ToolTip, { TooltipInstance } from '../Tooltip'

export interface DrawerPointsHandle {
  open: () => void
  close: () => void
  forceUpdate: () => void
  update: () => void
}

export interface DrawerPointsProps {
  onConfirm: (points: ChoicePoint[]) => void
}

const DrawerPoints = (props: DrawerPointsProps, ref: any) => {
  const { categories, onCreateCategory, choicePoints, updateData } = useContext(HyperloopContext)
  const { onConfirm } = props

  const pointsRef = useRef<ChoicePoint[]>([])
  const tooltipRef = useRef<TooltipInstance>(null)
  const [toolTipData, setToolTipData] = useState<any>({})
  const divRef = useRef<HTMLDivElement>(null)

  const [open, setOpen] = useState<boolean>(false)
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([])
  const forceUpdate = useForceUpdate()

  const columns: any[] = [
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
      dataIndex: 'label',
      width: 220,
      render: (col: string, record: any) => {
        const label = record.newLabel || col
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
            <Badge color={getColor(label)} text={label} />
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
      const item = { ...(pointsRef.current.find((item) => item.id === id)! as ChoicePoint) }
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

  useImperativeHandle<any, DrawerPointsHandle>(
    ref,
    () => {
      return {
        open: () => {
          setSelectedRowKeys(pointsRef.current.map((item) => item.id))
          setOpen(true)
        },
        close: onClose,
        forceUpdate,
        update: () => {
          pointsRef.current =
            choicePoints?.current.map((item) => {
              return { ...item, newLabel: updateData?.current[item.id] || '' }
            }) || []
        }
      }
    },
    []
  )

  useEffect(() => {
    if (choicePoints?.current) {
      pointsRef.current = choicePoints.current.map((item) => {
        return { ...item, newLabel: updateData?.current[item.id] || '' }
      })
    }
  }, [choicePoints?.current])

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
            <Space>Points List</Space>
            <span>
              ({selectedRowKeys.length} of {pointsRef.current.length} selected)
            </span>
            <Space>
              <Button type="primary" onClick={onClickConfirm}>
                Confirm
              </Button>
            </Space>
          </div>
        }
        size="large"
        open={open}
        placement="right"
        // mask={false}
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
          dataSource={pointsRef.current}
          pagination={false}
          scroll={{ y: 'calc(100vh - 130px)' }}
        />
      </Drawer>

      <div ref={divRef} style={{ display: 'none' }}>
        <div id="drawerTooltipWraper"></div>
      </div>

      <ToolTip
        ref={tooltipRef}
        elmId="drawerTooltipWraper"
        showText={false}
        categories={categories}
        data={toolTipData}
        onChange={onUpdate}
        onCreate={onCreate}
      />
    </>
  )
}

export default forwardRef<DrawerPointsHandle, DrawerPointsProps>(DrawerPoints)
