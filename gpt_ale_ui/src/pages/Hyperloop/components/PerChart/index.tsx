import React, { useContext, useEffect, useRef, useState } from 'react'

import { Empty } from 'antd'
import ReactECharts from 'echarts-for-react'

import { HyperloopContext } from '../HyperloopContext'

const PerChart = () => {
  const { loading, categories } = useContext(HyperloopContext)

  const [isEmpty, setEmpty] = useState<boolean>(false)

  const echartRef = useRef<any>()

  const option: any = {
    tooltip: {
      trigger: 'item',
      formatter: '{a} <br/>{b}: {c} ({d}%)'
    },
    grid: {
      left: '2%',
      right: '2%',
      bottom: '2%',
      top: '2%'
    },
    title: {
      text: '',
      top: '48%',
      left: '49%',
      textAlign: 'center'
    },
    series: [
      {
        type: 'pie',
        // selectedMode: 'single',
        radius: ['30%', '60%'],
        label: {
          formatter: '\n {title|{b}} \n {Value|{c}} ',
          rich: {
            title: {
              // color: '#eee',
              align: 'left'
            },
            Value: {
              height: 30,
              align: 'left'
            }
          }
        },
        tooltip: {
          formatter: '{b}: {c}'
        },
        itemStyle: {
          borderRadius: 4,
          borderColor: '#fff',
          borderWidth: 2
        },
        emphasis: {
          label: {
            show: true
          }
        },
        data: []
      }
    ]
  }

  useEffect(() => {
    if (!loading) {
      const echartInstance = echartRef.current.getEchartsInstance()
      option.title.text = `Total: ${(categories || []).reduce((total, item) => {
        total += item.value
        return total
      }, 0)}`
      option.series[0].data = categories
      echartInstance.setOption(option)
    }
  }, [loading, isEmpty, categories])

  return isEmpty ? (
    <Empty
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center'
      }}
      description="Categories will be displayed after first successful training"
    />
  ) : (
    <ReactECharts
      ref={echartRef}
      option={{}}
      showLoading={loading}
      style={{ minHeight: 312, height: '100%', width: '100%' }}
    />
  )
}

export default PerChart
