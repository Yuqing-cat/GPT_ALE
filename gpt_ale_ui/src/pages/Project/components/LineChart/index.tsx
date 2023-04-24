import React, { forwardRef, useContext, useEffect, useImperativeHandle, useRef } from 'react'

import ReactECharts from 'echarts-for-react'

import { getColor } from '@/utils/color'

export interface LineChartProps {
  loading: boolean
  data: any
}

export interface LineChartHandle {
  legendAllSelect: () => void
  legendInverseSelect: () => void
}

const LineChart = (props: LineChartProps, ref: any) => {
  const { loading, data } = props

  const echartRef = useRef<any>()
  useImperativeHandle<any, LineChartHandle>(ref, () => ({
    legendAllSelect() {
      const echart = echartRef.current.getEchartsInstance()
      echart.dispatchAction({
        type: 'legendAllSelect'
      })
    },
    legendInverseSelect() {
      const echart = echartRef.current.getEchartsInstance()
      echart.dispatchAction({
        type: 'legendInverseSelect'
      })
    }
  }))

  useEffect(() => {
    if (data) {
      const { global, ...rest } = data
      const echartInstance = echartRef.current.getEchartsInstance()
      const globalKeys = Object.keys(global)
      const xAxisData = global[globalKeys[0]].map((item: number[]) => {
        return item[0]
      })
      const legendData = Object.keys(rest).reduce(
        (list, name) => {
          list.push(name)
          return list
        },
        ['Global'] as string[]
      )
      const color = legendData.map((name) => getColor(name))
      const series = globalKeys.reduce((list, name, index) => {
        list.push({
          name: 'Global',
          type: 'line',
          xAxisIndex: index,
          yAxisIndex: index,
          // seriesLayoutBy: 'row',
          // smooth: true,
          data: global[name].map((value: number[]) => {
            return value[1]
          })
        })
        Object.keys(rest).forEach((category) => {
          list.push({
            name: category,
            type: 'line',
            xAxisIndex: index,
            yAxisIndex: index,
            // seriesLayoutBy: 'row',
            // smooth: true,
            data: rest[category][name].map((value: number[]) => {
              return value[1]
            })
          })
        })
        return list
      }, [] as any[])

      const option: any = {
        color,
        animationDuration: 2000,
        tooltip: {
          order: 'valueDesc',
          trigger: 'axis'
        },
        legend: {
          data: legendData
        },
        title: globalKeys.map((name, index) => {
          return {
            text: name,
            top: index < 2 ? '10%' : '54%',
            left: index % 2 === 0 ? '25%' : '75%',
            textAlign: 'center'
          }
        }),
        // visualMap: globalKeys.map((name, index) => {
        //   return {
        //     show: false,
        //     type: 'continuous',
        //     gridIndex: index,
        //     min: 0,
        //     max: 1
        //   }
        // }),
        grid: [
          { left: '7%', top: '14%', width: '38%', height: '36%' },
          { right: '7%', top: '14%', width: '38%', height: '36%' },
          { left: '7%', bottom: '5%', width: '38%', height: '36%' },
          { right: '7%', bottom: '5%', width: '38%', height: '36%' }
        ],
        xAxis: globalKeys.map((name, index) => {
          return {
            data: xAxisData,
            gridIndex: index
          }
        }),
        yAxis: globalKeys.map((name, index) => {
          return {
            gridIndex: index
          }
        }),
        series
      }
      echartInstance.setOption(option)
    }
  }, [data])

  return (
    <ReactECharts
      ref={echartRef}
      option={{}}
      showLoading={loading}
      style={{ minHeight: 600, height: 'calc(100vh - 260px)', width: '100%' }}
    />
  )
}

export default forwardRef<LineChartHandle, LineChartProps>(LineChart)
