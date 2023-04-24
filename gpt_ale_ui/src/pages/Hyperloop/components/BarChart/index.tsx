import React, { useContext, useEffect, useRef } from 'react'

import ReactECharts from 'echarts-for-react'

import { HyperloopContext } from '../HyperloopContext'

const BarChart = () => {
  const { loading, categories } = useContext(HyperloopContext)

  const echartRef = useRef<any>()

  useEffect(() => {
    if (categories?.length) {
      const echartInstance = echartRef.current.getEchartsInstance()
      const option: any = {
        color: [],
        legend: {
          data: []
        },
        grid: {
          left: '10%',
          right: '10%',
          bottom: '10%',
          top: '35%'
        },
        xAxis: [
          {
            type: 'category',
            data: ['category']
          }
        ],
        yAxis: [
          {
            type: 'value'
          }
        ],
        series: []
      }
      categories.forEach((item) => {
        option.color.push(item.itemStyle.color)
        option.legend.data.push(item.name)
        option.series.push({
          type: 'bar',
          name: item.name,
          data: [item.value]
        })
      })
      echartInstance.setOption(option)
    }
  }, [categories])

  return (
    <ReactECharts
      ref={echartRef}
      option={{}}
      showLoading={loading}
      style={{ minHeight: 312, height: '100%', width: '100%' }}
    />
  )
}

export default BarChart
