import React, {
  forwardRef,
  useContext,
  RefObject,
  useEffect,
  useImperativeHandle,
  useRef,
  useState
} from 'react'

import { PlusOutlined } from '@ant-design/icons'
import { Input, Select, Button, Typography, Divider } from 'antd'
import type { InputRef } from 'antd'

import { Category, ChoicePoint } from '@/api/palantir/interface'
import Portal from '@/components/Portal'

import { HyperloopContext } from '../HyperloopContext'

import styles from './index.module.less'

export interface TooltipInstance {
  show: () => void
  hide: () => void
  rect: () => DOMRectList
}

export interface TooltipProps {
  elmId?: string
  showText?: boolean
  elm?: RefObject<HTMLDivElement>
  data?: ChoicePoint
  content?: string
  position?: [number, number]
  categories?: Category[]
  value?: string
  onCreate?: (category: string) => void
  onChange?: (data: ChoicePoint, name: string) => void
}
const { Paragraph } = Typography

const Tooltip = (props: TooltipProps, ref: any) => {
  const { updateData } = useContext(HyperloopContext)
  const { elmId, data, showText = true, categories, onCreate, onChange } = props

  const elmRef = useRef(document.createElement('div'))

  const echartTooltipElementRef = useRef<HTMLElement | null>()
  const echartElementRef = useRef<HTMLElement | null>()
  const tooltipRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<InputRef>(null)

  const [currentValue, setCurrentValue] = useState(
    updateData?.current[data!.id] || data?.label || ''
  )

  const [options, setOptions] = useState<{ label: string; value: string }[]>([])

  const [name, setName] = useState('')

  const onNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setName(e.target.value)
  }

  const onAddCategory = () => {
    const newCategory = name.trim()

    if (newCategory && categories?.findIndex((item) => item.name === newCategory) === -1) {
      setName('')
      setCurrentValue(newCategory)

      onCreate?.(newCategory)
    }
  }

  const onSelectChange = (value: string) => {
    setCurrentValue(value)
  }

  const onClickConfirm = () => {
    elmRef.current.style.display = 'none'

    setTimeout(() => {
      onChange?.(data!, currentValue)
    }, 0)
  }

  useEffect(() => {
    const options = categories?.map(({ name }) => ({ label: name, value: name })) || []
    setOptions(options)
  }, [categories])

  useEffect(() => {
    if (!echartTooltipElementRef.current) {
      echartTooltipElementRef.current = elmId
        ? document.getElementById(elmId)
        : document.getElementById('tool-tip')?.parentElement
      // echartTooltipElementRef.current?.setAttribute('id', 'echart-tooltip')
      echartElementRef.current = echartTooltipElementRef.current?.parentElement
      echartElementRef.current?.appendChild(elmRef.current)
    }

    setTimeout(() => {
      setCurrentValue(updateData?.current[data!.id] || data?.label || '')

      elmRef.current.style.display = 'block'
      elmRef.current.style.setProperty(
        'border-color',
        echartTooltipElementRef.current?.style.borderColor || ''
      )
      elmRef.current.style.setProperty(
        'transform',
        echartTooltipElementRef.current?.style.transform || ''
      )
    }, 0)
  }, [elmId, data])

  useEffect(() => {
    if (
      categories?.findIndex(({ name }) => {
        return name === data?.gpt
      }) === -1
    ) {
      setName(data?.gpt !== 'unknown' ? data?.gpt || '' : '')
    } else {
      setName('')
    }
  }, [data, categories])

  useEffect(() => {
    elmRef.current.id = 'custom-tooltip-wrap'
    const documentMousedownHandle = (e: MouseEvent) => {
      elmRef.current.style.display = 'none'
    }
    const elmMousedownHandle = (e: MouseEvent) => {
      e.stopPropagation()
    }

    document.addEventListener('mousedown', documentMousedownHandle, false)
    elmRef.current.addEventListener('mousedown', elmMousedownHandle, false)

    return () => {
      document.removeEventListener('mousedown', documentMousedownHandle, false)
      elmRef.current.removeEventListener('mousedown', elmMousedownHandle, false)
    }
  }, [])

  useImperativeHandle<any, TooltipInstance>(
    ref,
    () => {
      return {
        show: () => {
          elmRef.current.style.display = 'block'
        },
        hide: () => {
          elmRef.current.style.display = 'none'
        },
        rect: () => {
          return tooltipRef.current!.getClientRects()
        }
      }
    },
    []
  )

  return (
    <Portal getContainer={() => elmRef.current as Element}>
      <div key="tooltip" ref={tooltipRef} className={styles.tooltip}>
        {showText && (
          <Paragraph style={{ whiteSpace: 'break-spaces', maxHeight: 90, overflow: 'auto' }}>
            {data?.text}
          </Paragraph>
        )}
        <div className={styles.body}>
          <Select
            open
            dropdownRender={(menu) => (
              <>
                {menu}
                <Divider style={{ margin: '8px 0' }} />
                <div className={styles.addBar}>
                  <span>GPT3: </span>
                  <Input
                    ref={inputRef}
                    placeholder="Please enter category"
                    maxLength={25}
                    value={name}
                    onChange={onNameChange}
                  />
                  <Button icon={<PlusOutlined />} onClick={onAddCategory}>
                    Add
                  </Button>
                </div>
              </>
            )}
            getPopupContainer={() => {
              return tooltipRef.current!
            }}
            options={options}
            placeholder="custom dropdown render"
            popupClassName={styles.blockPopup}
            style={{ width: '100%' }}
            value={currentValue}
            virtual={false}
            onChange={onSelectChange}
          />
          <Button type="primary" onClick={onClickConfirm}>
            Confirm
          </Button>
        </div>
      </div>
    </Portal>
  )
}

export default forwardRef<TooltipInstance, TooltipProps>(Tooltip)
