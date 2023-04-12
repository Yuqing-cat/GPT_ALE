import React, { useContext, useEffect, useRef, useState } from 'react'

import { Card, Empty, Radio } from 'antd'
import type { RadioChangeEvent } from 'antd'
import ReactECharts from 'echarts-for-react'

import { Point } from '@/api/palantir/interface'

import { HyperloopContext } from '../HyperloopContext'

const HeatmapChart = () => {
  const { loading, categorieNames, occurance } = useContext(HyperloopContext)

  const echartRef = useRef<any>()

  const [isEmpty, setIsEmpty] = useState<boolean>(false)

  const [value, setValue] = useState<string>('true')

  const onChange = ({ target: { value } }: RadioChangeEvent) => {
    setValue(value)
  }

  useEffect(() => {
    if (!loading && occurance?.current[value]) {
      const max = occurance?.current[value].reduce((max: number, item: Point) => {
        if (item[2] > max) {
          return item[2]
        }
        return max
      }, 0)

      const echartInstance = echartRef.current.getEchartsInstance()
      const option = {
        tooltip: {
          position: 'left'
        },
        grid: {
          left: '100',
          right: '10%',
          bottom: '15%',
          top: '10%'
        },
        xAxis: {
          name: 'Truth',
          type: 'category',
          data: categorieNames?.current,
          splitArea: {
            show: true
          },
          axisLabel: {
            show: false,
            width: '80',
            overflow: 'truncate',
            interval: 0,
            rotate: 45
          }
        },
        yAxis: {
          name: 'Prediction',
          type: 'category',
          data: categorieNames?.current,
          splitArea: {
            show: true
          },
          axisLabel: {
            width: '80',
            overflow: 'truncate'
          }
        },
        visualMap: {
          min: value === 'none' ? 1 : 0,
          max: value === 'none' ? max : 1,
          calculable: true,
          splitNumber: 8,
          left: 'right',
          top: 'center',
          orient: 'vertical'
          // inRange: {
          //   color: ['#efdfcf', '#f5c09e', '#ed4b40', '#94215a', '#3c1b42', '#010215']
          // }
        },
        series: [
          {
            type: 'heatmap',
            data: occurance?.current[value],
            // label: {
            //   show: true
            // },
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowColor: 'rgba(0, 0, 0, 0.5)'
              }
            }
          }
        ]
      }
      echartInstance.setOption(option)
    } else if (!loading && !occurance?.current.length) {
      setIsEmpty(true)
    }
  }, [loading, occurance?.current, value])

  return (
    <Card
      extra={
        occurance?.current.all ? (
          <Radio.Group
            options={[
              {
                label: 'None',
                value: 'none'
              },
              {
                label: 'All',
                value: 'all'
              },
              {
                label: 'Pred',
                value: 'pred'
              },
              {
                label: 'True',
                value: 'true'
              }
            ]}
            value={value}
            optionType="button"
            onChange={onChange}
          />
        ) : null
      }
      id="guide-card-3"
      title="Confusion matrix"
      style={{ flex: 1, display: 'flex', flexDirection: 'column' }}
      bodyStyle={{ padding: 0, flex: 1 }}
    >
      {isEmpty ? (
        <Empty
          style={{
            minHeight: 312,
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center'
          }}
          description="Model accuracy will be displayed after first successful training"
        />
      ) : (
        <ReactECharts
          ref={echartRef}
          option={{}}
          showLoading={loading}
          style={{ minHeight: 312, height: '100%', width: '100%' }}
        />
      )}
    </Card>
  )
}

export default HeatmapChart
