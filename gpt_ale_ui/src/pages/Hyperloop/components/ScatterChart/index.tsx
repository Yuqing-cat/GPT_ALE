import React, {
  forwardRef,
  useContext,
  useEffect,
  useImperativeHandle,
  useRef,
  useState
} from 'react'

import { Empty } from 'antd'
import ReactECharts from 'echarts-for-react'
import { chunk, flatten, isEmpty } from 'lodash-es'

import { ChoicePoint, Point } from '@/api/gpt_ale/interface'
import { getColor } from '@/utils/color'

import { HyperloopContext } from '../HyperloopContext'
import ToolTip, { TooltipInstance } from '../Tooltip'

export interface ScatterChartProps {
  getPoint: (point: number[]) => void
  className?: string
}
export interface ScatterChartHandle {
  showTooltip: (id: number) => void
  unBrush: () => void
  getPoint: (id: number) => void
}

const ScatterChart = (props: ScatterChartProps, ref: any) => {
  const {
    loading,
    isFianl,
    points,
    choicePoints,
    categories,
    updateData,
    isStartHyperloop,
    transformations,
    onCreateCategory,
    onChangeCategory,
    onBrushselected,
    onBrushEnd
  } = useContext(HyperloopContext)

  const { getPoint, className } = props

  const [toolTipData, setToolTipData] = useState<any>({})

  const [isEmpty, setIsEmpty] = useState<boolean>(false)

  const toolTipRef = useRef<TooltipInstance>(null)
  const isShowTooltipRef = useRef(true)

  const echartRef = useRef<any>()

  const optionRef = useRef<any>({
    darkMode: true,
    xAxis: {
      scale: true,
      show: false
    },
    yAxis: {
      scale: true,
      show: false
    },
    grid: {
      left: 20,
      right: 20,
      bottom: 20,
      top: 20
    },
    brush: {
      // brushLink: 'all',
      toolbox: ['rect', 'polygon', 'lineX', 'lineY', 'keep', 'clear']
    },
    tooltip: {
      trigger: 'item',
      triggerOn: 'click',
      enterable: true,
      formatter: ({ data }: any) => {
        if (!isStartHyperloop?.current && isShowTooltipRef.current) {
          const point = data[2] as any
          setToolTipData({ ...point })
        }
        return '<div id="tool-tip"></div>'
      }
    },

    position: (point: number[], params: any, dom: HTMLElement, rect: any, size: any) => {
      let [left, top] = point
      const rectlist = toolTipRef.current?.rect()
      let height = 300
      if (rectlist!.length > 0) {
        height = rectlist!.item(0)!.height + 20
      }

      if (height + top > size.viewSize[1]) {
        top -= height
      } else {
        top -= 10
      }
      if (top < 50) {
        top = 50
      }
      if (360 + left > size.viewSize[0]) {
        left -= 360
      } else {
        left += 20
      }
      getPoint(point)
      return [left, top]
    },
    dataZoom: [
      {
        type: 'inside'
      },
      {
        type: 'inside',
        orient: 'vertical'
      }
    ]
  })

  const onUpdate = (point: ChoicePoint, name: string) => {
    onChangeCategory?.(point, name)
    const echartInstance = echartRef.current.getEchartsInstance()
    const option = echartInstance.getOption()
    echartInstance.setOption(option, { replaceMerge: ['series'] })
  }

  const onCreate = (category: string) => {
    onCreateCategory?.([category])
    onUpdate(toolTipData, category)
  }

  useImperativeHandle<any, ScatterChartHandle>(ref, () => ({
    showTooltip(index: number) {
      const echartInstance = echartRef.current.getEchartsInstance()

      echartInstance.dispatchAction({
        type: 'showTip',
        seriesIndex: optionRef.current.series.length - 1,
        dataIndex: index,
        position: optionRef.current.position
      })
      echartInstance.dispatchAction({
        type: 'select',
        seriesIndex: optionRef.current.series.length - 1,
        dataIndex: index
      })
    },
    unBrush: () => {
      const echartInstance = echartRef.current.getEchartsInstance()
      const option = echartInstance.getOption()
      echartInstance.dispatchAction({
        type: 'brush',
        areas: []
      })

      echartInstance.setOption(option, { replaceMerge: ['series'] })
    },
    getPoint: (index: number) => {
      const echartInstance = echartRef.current.getEchartsInstance()
      isShowTooltipRef.current = false
      echartInstance.dispatchAction({
        type: 'showTip',
        seriesIndex: optionRef.current.series.length - 1,
        dataIndex: index,
        position: optionRef.current.position
      })
      isShowTooltipRef.current = true
    }
  }))

  useEffect(() => {
    if (!loading) {
      const echartInstance = echartRef.current.getEchartsInstance()
      const series: any[] = (optionRef.current.series = [])
      if (isFianl) {
        const all_points = points?.current as Record<string, Point[]>
        const keys = Object.keys(all_points)
        keys.forEach((name) => {
          const points = all_points[name]

          series.push({
            name,
            type: 'scatter',
            itemStyle: {
              color: getColor(name)
              // opacity: 0.5
            },
            tooltip: {
              trigger: 'none'
            },
            cursor: 'auto',
            emphasis: {
              disabled: true
            },
            large: true,
            largeThreshold: 8000,
            symbolSize: 10,
            dimensions: ['x', 'y'],
            data: new Int16Array(flatten(points))
          })

          echartInstance.setOption(optionRef.current)
        })

        setIsEmpty(keys.length === 0)
      } else {
        const transformationsKey = transformations?.current.reduce((keys: any, item) => {
          keys[item.index] = item
          return keys
        }, {} as any)
        const span = document.createElement('span')
        const all_points = points?.current as ChoicePoint[]

        const transformPoints1: any[] = []
        const transformPoints2: any[] = []
        const staticPoint1: any[] = []
        const staticPoint2: any[] = []

        all_points.forEach((item) => {
          const { point } = item
          const choice = transformationsKey[item.id]
          if (choice) {
            transformPoints1.push([...choice.old_loc, item])
          } else {
            staticPoint1.push([...point, item])
          }
        })

        choicePoints?.current.map((item) => {
          const { point } = item
          const choice = transformationsKey[item.id]
          if (choice) {
            transformPoints2.push([...choice.old_loc, item])
          } else {
            staticPoint2.push([...point, item])
          }
        })

        series.push({
          name: 'Points',
          type: 'scatter',
          zlevel: 10,
          symbolSize: 10,
          itemStyle: {
            // borderWidth: 2,
            // borderColor: '#000',
            opacity: 0.5,
            color: ({ data }: any) => {
              const { label, id, score } = data[2]
              return getColor(updateData?.current[id] || label, score)
            }
          },
          symbol: (value: any[]) => {
            if (updateData?.current[value[2].id]) {
              return 'path://M23.6 2c-3.363 0-6.258 2.736-7.599 5.594-1.342-2.858-4.237-5.594-7.601-5.594-4.637 0-8.4 3.764-8.4 8.401 0 9.433 9.516 11.906 16.001 21.232 6.13-9.268 15.999-12.1 15.999-21.232 0-4.637-3.763-8.401-8.4-8.401z'
            }
            return 'circle'
          },
          emphasis: {
            disabled: true
          },
          selectedMode: 'single',
          select: {
            // disabled: true,
            itemStyle: {
              shadowBlur: 0,
              borderWidth: 2,
              borderColor: '#1890ff'
            }
          },
          animation: false,
          animationDurationUpdate: 3000,
          animationDuration: 4000,
          data: transformPoints1
        })

        series.push({
          name: 'Points',
          type: 'scatter',
          zlevel: 100,
          symbolSize: 15,
          itemStyle: {
            borderWidth: 2,
            borderColor: '#000',
            color: ({ data }: any) => {
              const { label, id, score } = data[2]
              return getColor(updateData?.current[id] || label, score)
            }
          },
          symbol: (value: any[]) => {
            if (updateData?.current[value[2].id]) {
              return 'path://M23.6 2c-3.363 0-6.258 2.736-7.599 5.594-1.342-2.858-4.237-5.594-7.601-5.594-4.637 0-8.4 3.764-8.4 8.401 0 9.433 9.516 11.906 16.001 21.232 6.13-9.268 15.999-12.1 15.999-21.232 0-4.637-3.763-8.401-8.4-8.401z'
            }
            return 'circle'
          },
          emphasis: {
            disabled: true
          },
          selectedMode: 'single',
          select: {
            // disabled: true,
            itemStyle: {
              shadowBlur: 0,
              borderWidth: 2,
              borderColor: '#1890ff'
            }
          },
          animation: false,
          animationDurationUpdate: 3000,
          animationDuration: 4000,
          data: transformPoints2
        })

        const seriesIndex = series.length - 1

        echartInstance.setOption(optionRef.current)
        setTimeout(() => {
          optionRef.current.series.forEach((serie: any, index: number) => {
            serie.animation = true
            if (index === seriesIndex) {
              serie.data = transformPoints2
                .map((item: any[]) => {
                  const { id } = item[2]
                  const choice = transformationsKey[id]
                  return [...choice.new_loc, item[2]]
                })
                .concat(staticPoint2)
            } else {
              serie.data = transformPoints1
                .map((item: any[]) => {
                  const { id } = item[2]
                  const choice = transformationsKey[id]
                  return [...choice.new_loc, item[2]]
                })
                .concat(staticPoint1)
            }
          })

          echartInstance.setOption(optionRef.current)
          setTimeout(() => {
            optionRef.current.series.forEach((serie: any) => {
              serie.animation = false
            })

            echartInstance.setOption(optionRef.current)
          }, 4000)
        }, 200)

        echartInstance.on('mousemove', (params: any) => {
          const { data, event } = params
          const { id, label } = data[2] as any
          const tooltip = document.querySelector('#tooltip')

          span.innerText = updateData?.current[id] || label
          tooltip?.append(span)
          tooltip?.setAttribute(
            'style',
            `position: absolute; 
            z-index: 9999999;
            white-space: nowrap;
            box-shadow: rgb(0 0 0 / 20%) 1px 2px 10px;
            border-width: 1px;
            border-style: solid;
            border-radius: 4px;
            padding: 8px 10px;
            top: 0px;
            left: 0px;
            background: #fff;
            border-color: ${getColor(span.innerText)};
            transform: translate3d(${event.offsetX}px, ${event.offsetY - 50}px, 0px);
            `
          )
        })
        echartInstance.on('mouseout', () => {
          const tooltip = document.querySelector('#tooltip')
          tooltip?.setAttribute('style', 'display:none')
        })
      }
    }
  }, [loading, isFianl, points?.current, choicePoints?.current])

  useEffect(() => {
    const echartInstance = echartRef.current.getEchartsInstance()
    echartInstance.on('mousedown', 'series', (params: any) => {
      const { seriesName, event } = params
      if (seriesName === 'Points') {
        event.event.stopPropagation()
      }
    })

    echartInstance.on('brushselected', (params: any) => {
      const option = echartInstance.getOption()
      onBrushselected?.(params, option)
    })
    echartInstance.on('brushEnd', onBrushEnd)
  }, [])

  useEffect(() => {
    if (Object.keys(updateData?.current || {}).length) {
      const echartInstance = echartRef.current.getEchartsInstance()
      echartInstance.setOption(echartInstance.getOption())
    }
  }, [Object.keys(updateData?.current || {}).length])

  return (
    <>
      {isEmpty ? (
        <Empty
          style={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center'
          }}
          description="Scatter will be displayed after first successful training"
        />
      ) : (
        <ReactECharts
          ref={echartRef}
          className={className}
          showLoading={loading}
          option={{}}
          style={{ minHeight: 760, height: '100%', width: '100%' }}
        />
      )}
      <div id="tooltip" style={{ display: 'none' }}></div>
      <ToolTip
        ref={toolTipRef}
        categories={categories}
        data={toolTipData}
        onChange={onUpdate}
        onCreate={onCreate}
      />
    </>
  )
}

export default forwardRef<ScatterChartHandle, ScatterChartProps>(ScatterChart)
